

## Point transform
$$
\Large
X_{world}
\xrightarrow{\text{View (Extrinsic)}} X_{camera}
\xrightarrow{\text{Projection (not Intrinsic)}} X_{clip}
\xrightarrow{\text{Perspective Divide}} X_{NDC}
\xrightarrow{\text{Viewport}} X_{screen}
$$


### 🔹 Step-by-Step Explanation

| Stage | Operation | Mathematical Form | Description |
|:--|:--|:--|:--|
| World → Camera | Apply extrinsic parameters | $X_{camera} = R\,X_{world} + t$ | Transforms 3D points from world to camera coordinates using rotation $R$ and translation $t$ |
| Camera → Clip Space | Apply projection matrix | $X_{clip} = P\,\tilde{X}_{camera}$ | Converts camera coordinates to clip space using a 4×4 projection matrix (often perspective) |
| Clip → NDC | Perspective divide | $X_{ndc} = X_{clip} / w_{clip}$ | Divides by the homogeneous coordinate $w$ to normalize into the range $[-1,1]$ |
| NDC → Screen | Viewport transform | $X_{screen} = \big((x_{ndc}+1)\tfrac{W}{2},\; (1-y_{ndc})\tfrac{H}{2}\big)$ | Maps normalized coordinates to screen pixels using image width $W$ and height $H$ |

$P =
\begin{bmatrix}
\dfrac{f}{a} & 0 & 0 & 0 \\[6pt]
0 & f & 0 & 0 \\[6pt]
0 & 0 & \dfrac{f_a + n}{\,n - f_a\,} & \dfrac{2 f_a n}{\,n - f_a\,} \\[6pt]
0 & 0 & -1 & 0
\end{bmatrix}$

---

## Covariance transform

$$
\Large
\Sigma_{world}
\xrightarrow{\text{View (Extrinsic rotation only)}} 
\Sigma_{camera}
\xrightarrow{\text{Projection Jacobian (related to Intrinsic)}} 
\Sigma_{2D}
$$



### 🔹 Step-by-Step Explanation

| Stage | Operation | Mathematical Form | Description |
|:--|:--|:--|:--|
| World → Camera | Extrinsic rotation | $\Sigma_{camera} = R\,\Sigma_{world}\,R^{T}$ | Translation does not affect variance; only rotation is applied |
| Camera → Image Plane | Local linearization of perspective projection | $J = \begin{bmatrix} f_x/z & 0 & -f_x x / z^2 \\[4pt] 0 & f_y/z & -f_y y / z^2 \end{bmatrix}$ | Derived by taking partial derivatives of $u = f_x x/z$ and $v = f_y y/z$ |
| Image-plane covariance | Covariance propagation | $\Sigma_{2D} = J\,\Sigma_{camera}\,J^{T}$ | Produces the 2D elliptical shape and size on the screen |


$P_{intrinsic} =
\begin{bmatrix}
f_x & 0   & c_x & 0 \\[6pt]
0   & f_y & c_y & 0 \\[6pt]
0   & 0   & 1   & 0
\end{bmatrix}$

---

### 🧩 Overall Formula Summary

$$
\Large
\Sigma_{2D} = J\,R\,\Sigma_{world}\,R^{T}\,J^{T}
$$

### 🧩 Jacobian Explanation

#### 1️⃣ Projection Function (includes Intrinsic)

$$
\begin{cases}
u = f_x \dfrac{x_c}{z_c} + c_x \\[6pt]
v = f_y \dfrac{y_c}{z_c} + c_y
\end{cases}
\
$$

Here, $f_x, f_y, c_x, c_y$ are the **camera intrinsics**.

---

#### 2️⃣ Definition of the Jacobian

For the mapping $(x_c, y_c, z_c) \rightarrow (u, v)$,  
take the partial derivatives:

$$
J \;=\;
\dfrac{\partial(u, v)}{\partial(x_c, y_c, z_c)} \;=\;
\begin{bmatrix}
\dfrac{\partial u}{\partial x_c} & 
\dfrac{\partial u}{\partial y_c} & 
\dfrac{\partial u}{\partial z_c} \\[6pt]
\dfrac{\partial v}{\partial x_c} & 
\dfrac{\partial v}{\partial y_c} & 
\dfrac{\partial v}{\partial z_c}
\end{bmatrix}
\;=\;
\begin{bmatrix}
\dfrac{f_x}{z_c} & 0 & -\dfrac{f_x x_c}{z_c^2} \\[6pt]
0 & \dfrac{f_y}{z_c} & -\dfrac{f_y y_c}{z_c^2}
\end{bmatrix}
$$


---

