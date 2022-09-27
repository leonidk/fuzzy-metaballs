import jax
import jax.numpy as jnp

hyperparams_models = [-2.78, -0.1,  6.4, -5.44]
hyperparams_kitti = [-5.75, 0.476, 7.16, -7.45] 
hyperparams_blend = [-4.5, 0.09, 6.24, -5.36] 
hyperparams = hyperparams_models

def jax_stable_exp(z,s=1,axis=0):
    z = s*z
    z = z- z.max(axis)
    z = jnp.exp(z)
    return z

def render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3, beta_4, beta_5):
    prec = jnp.triu(prec_full)
    weights = jnp.exp(weights_log)
    weights = weights/weights.sum()

    def perf_idx(prcI,w,meansI):
        prc = prcI.T
        #prc = jnp.diag(jnp.sign(jnp.diag(prc))) @ prc
        div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20

        def perf_ray(r_t):
            r = r_t[0]
            t = r_t[1]
            p =  meansI -t 

            vsv = ((prc @ r)**2).sum()
            psv = ((prc @ p) * (prc@r)).sum()

            # linear
            res = (psv)/(vsv)
            
            v = r * res - p

            d0 = ((prc @ v)**2).sum()
            d2 = -0.5*d0 + jnp.log(w)
            #d3 =  d2 + jnp.log(div)

            return res,d2
        res,d2  = jax.vmap((perf_ray))(camera_starts_rays) # jit perf
        return res, d2

    zs,stds = jax.vmap(perf_idx)(prec,weights,means)  # jit perf
    sig1 = (zs > 0)# sigmoid

    w = sig1*jnp.nan_to_num(jax_stable_exp(-zs*beta_2 + beta_3*stds))+1e-20

    wgt  = w.sum(0)
    init_t=  (w*jnp.nan_to_num(zs)).sum(0)/jnp.where(wgt==0,1,wgt)
    # est_alpha = 1-jnp.exp(-beta_4*(jnp.exp(stds).sum(0)) ) # simplier but splottier
    est_alpha = jnp.tanh(beta_4*(jnp.exp(stds).sum(0)+beta_5) )*0.5 + 0.5 # more complex but flatter


    return init_t,stds,est_alpha


def render_func(means, prec_full, weights_log, camera_rays, axangl, t, beta_2, beta_3, beta_4, beta_5):
    scale = jnp.sqrt(axangl @ axangl)
    vec = axangl/scale
    Rx = jnp.array([[0,0,0],[0,0,-1.0],[0,1,0]])
    Ry = jnp.array([[0,0,1],[0,0,0],[-1.0,0,0]])
    Rz = jnp.array([[0,-1.0,0],[1,0,0],[0,0,0]])
    Rmat = jnp.array([Rx,Ry,Rz])
    K = Rmat @ vec
    ctheta = jnp.cos(scale)
    stheta = jnp.sin(scale)
    Rest = jnp.where(scale > 1e-12,(jnp.eye(3) + stheta*K + (1-ctheta)*(K@K)), jnp.eye(3)).T
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3, beta_4, beta_5)