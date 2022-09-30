import numpy as np



def patch_ex(dest_img, src_img):
    dest_img = dest_img.copy()
    src_img = src_img.copy()
    size = np.random.randint(src_img.shape[0]*0.15, src_img.shape[1]*0.2)
    cut_center = np.random.randint(src_img.shape[0]/2-size, src_img.shape[0]/2)
    patch = src_img[cut_center:(cut_center+size),cut_center:(cut_center+size)]
  
    
    center_bound = 0.3
    center1 = np.random.randint(dest_img.shape[0]*center_bound-size, dest_img.shape[0]*(1-center_bound))
    center2 = np.random.randint(dest_img.shape[0]*center_bound-size, dest_img.shape[0]*(1-center_bound))

    dest_img[center1:center1+size,center2:center2+size] = patch

    return dest_img