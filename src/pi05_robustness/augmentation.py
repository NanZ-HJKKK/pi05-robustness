"""Training-only JAX augmentation on images in [0,1]. Imports are lazy.

Legacy view/view_drop affect the main view. Registered view_wrist_drop applies
geometry to half of main images and independently drops 15% of wrist images.
Default preprocessing and inference bypass extra augmentation. Masks stay valid.
"""
import os


def homography(src, dst, xp):
    """Solve src -> dst using eight inhomogeneous DLT equations."""
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    z, o = xp.zeros_like(x), xp.ones_like(x)
    a = xp.stack((xp.stack((x,y,o,z,z,z,-u*x,-u*y), axis=-1),
                  xp.stack((z,z,z,x,y,o,-v*x,-v*y), axis=-1)), axis=1).reshape(8,8)
    b = xp.stack((u,v), axis=1).reshape(8)
    h = xp.linalg.solve(a,b)
    return xp.concatenate((h, xp.ones(1, dtype=h.dtype))).reshape(3,3)


def random_perspective(key, image, magnitude=0.10):
    import jax
    import jax.numpy as jnp
    from jax.scipy.ndimage import map_coordinates
    height, width = image.shape[:2]
    corners = jnp.array([[0.,0.],[width-1.,0.],[width-1.,height-1.],[0.,height-1.]], dtype=jnp.float32)
    offset = jax.random.uniform(key, (4,2), minval=-magnitude, maxval=magnitude) * jnp.array([width-1,height-1])
    # Inverse mapping: output (displaced corner) -> input (original corner).
    inverse = homography(corners + offset, corners, jnp)
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    grid = jnp.stack((xx.ravel(), yy.ravel(), jnp.ones(height*width)))
    sample = inverse @ grid
    denom = sample[2]
    denom = jnp.where(jnp.abs(denom) < 1e-6, jnp.where(denom < 0, -1e-6, 1e-6), denom)
    coords = jnp.stack((sample[1]/denom, sample[0]/denom))
    warped = jax.vmap(lambda channel: map_coordinates(channel, coords, order=1, mode="nearest"), in_axes=2, out_axes=1)(image)
    return jnp.clip(warped.reshape(height,width,image.shape[-1]),0,1)


def camera_transform(key, image):
    """Approximate camera geometry in image space; actions stay in world coordinates."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.ndimage import map_coordinates
    angle_key,scale_key,perspective_key=jax.random.split(key,3)
    height,width=image.shape[:2]
    corners=jnp.array([[0.,0.],[width-1.,0.],[width-1.,height-1.],[0.,height-1.]],dtype=jnp.float32)
    center=jnp.array([(width-1)/2,(height-1)/2])
    angle=jax.random.uniform(angle_key,(),minval=-25.,maxval=25.)*jnp.pi/180
    scale=jax.random.uniform(scale_key,(),minval=.65,maxval=1.35)
    rotation=jnp.array([[jnp.cos(angle),-jnp.sin(angle)],[jnp.sin(angle),jnp.cos(angle)]])
    offsets=jax.random.uniform(perspective_key,(4,2),minval=-.08,maxval=.08)*jnp.array([width-1,height-1])
    destination=(corners-center)@rotation.T*scale+center+offsets
    inverse=homography(destination,corners,jnp)
    yy,xx=jnp.meshgrid(jnp.arange(height),jnp.arange(width),indexing='ij')
    points=inverse@jnp.stack([xx.ravel(),yy.ravel(),jnp.ones(height*width)])
    denominator=jnp.where(jnp.abs(points[2])<1e-6,1e-6,points[2])
    coordinates=jnp.stack([points[1]/denominator,points[0]/denominator])
    pixels=jax.vmap(lambda c:map_coordinates(c,coordinates,order=1,mode='nearest'),in_axes=2,out_axes=1)(image)
    return jnp.clip(pixels.reshape(height,width,3),0,1)


def augment_batch(key, images, mode=None, camera="main"):
    import jax
    import jax.numpy as jnp
    mode = os.environ.get("OPENPI_AUG", "default") if mode is None else mode
    if mode == "default":
        return images
    if mode == "view_wrist_drop":
        if camera == "unused":
            return images
        transform_key,selection_key=jax.random.split(key)
        if camera == "main":
            transformed=jax.vmap(camera_transform)(jax.random.split(transform_key,len(images)),images)
            selected=jax.random.bernoulli(selection_key,.5,(len(images),1,1,1))
            return jnp.where(selected,transformed,images)
        if camera == "wrist":
            dropped=jax.random.bernoulli(selection_key,.15,(len(images),1,1,1))
            return jnp.where(dropped,jnp.zeros_like(images),images)
        raise ValueError("Unsupported camera: "+camera)
    if mode not in ("view", "view_drop"):
        raise ValueError("Unsupported OPENPI_AUG: " + mode)
    warp_key, drop_key = jax.random.split(key)
    warped = jax.vmap(random_perspective)(jax.random.split(warp_key, len(images)), images)
    if mode == "view_drop":
        drop = jax.random.bernoulli(drop_key, 0.15, (len(images),1,1,1))
        warped = jnp.where(drop, jnp.zeros_like(warped), warped)
    return warped
