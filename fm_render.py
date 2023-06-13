import jax
import jax.numpy as jnp

hyperparams = [2.0, 0.25]

def jax_stable_exp(z,s=1,axis=0):
    z = s*z
    z = z- z.max(axis)
    z = jnp.exp(z)
    return z

def mrp_to_rot(vec):
    vec_mag = vec @ vec
    vec_mag_num = (1-vec_mag)
    vec_mag_den = ((1+vec_mag)**2)
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    R1 = jnp.eye(3) - ( ((4*vec_mag_num)/vec_mag_den) * K) + ((8/vec_mag_den) * (K @ K))
    R2 = jnp.eye(3)

    Rest = jnp.where(vec_mag > 1e-12,R1,R2)
    return Rest

def axangle_to_rot(axangl):
    scale = jnp.sqrt(axangl @ axangl)
    vec = axangl/scale
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    ctheta = jnp.cos(scale)
    stheta = jnp.sin(scale)
    R1 = jnp.eye(3) + stheta*K + (1-ctheta)*(K @ K)
    R2 = jnp.eye(3)
    Rest = jnp.where(scale > 1e-12,R1.T, R2)
    return Rest

def quat_to_rot(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    R1 = jnp.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    R2 = jnp.eye(3)
    return jnp.where(Nq > 1e-12,R1,R2)


def render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3):
    prec = jnp.triu(prec_full)
    # weights = jnp.exp(weights_log)
    # weights = weights/weights.sum()

    def perf_idx(prcI,w,meansI):
        prc = prcI.T
        #div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20

        def perf_ray(r_t):
            r = r_t[0]
            t = r_t[1]
            p =  meansI -t 

            projp = prc @ p
            vsv = ((prc @ r)**2).sum()
            psv = ((projp) * (prc@r)).sum()

            # linear
            res = (psv)/(vsv)
            
            v = r * res - p

            d0 = ((prc @ v)**2).sum()
            d2 = -0.5*d0 + w
            #d3 =  d2 + jnp.log(div)

            norm_est = projp/jnp.linalg.norm(projp)
            norm_est = jnp.where(r@norm_est < 0,norm_est,-norm_est)
            return res,d2,norm_est
        return jax.vmap((perf_ray))(camera_starts_rays) 

    zs,stds,projp = jax.vmap(perf_idx)(prec,weights_log,means)  # jit perf

    est_alpha = 1-jnp.exp(-jnp.exp(stds).sum(0) ) # simplier but splottier
    sig1 = (zs > 0)# sigmoid
    w = sig1*jnp.nan_to_num(jax_stable_exp(-zs*beta_2 + beta_3*stds))+1e-20

    wgt  = w.sum(0)
    div = jnp.where(wgt==0,1,wgt)
    w = w/div

    init_t=  (w*jnp.nan_to_num(zs)).sum(0)
    est_norm = (projp * w[:,:,None]).sum(axis=0)
    est_norm = est_norm/jnp.linalg.norm(est_norm,axis=1,keepdims=True)

    return init_t,est_alpha,est_norm,w

# axis angle rotations n * theta
def render_func_axangle(means, prec_full, weights_log, camera_rays, axangl, t, beta_2, beta_3):
    Rest = axangle_to_rot(axangl)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# modified rod. parameters n * tan(theta/4)
def render_func_mrp(means, prec_full, weights_log, camera_rays, mrp, t, beta_2, beta_3):
    Rest = mrp_to_rot(mrp)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# quaternions [cos(theta/2), sin(theta/2) * n]
def render_func_quat(means, prec_full, weights_log, camera_rays, quat, t, beta_2, beta_3):
    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)