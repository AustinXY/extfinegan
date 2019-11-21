**Version 2.6.1.1**

added soft constraints (concentration loss; separation loss, parent child similarity loss, equivariance loss)<br>

implemented mask incomplete loss: <br>
using new discriminator;<br>
discriminator uses complete image as real sample;<br>

equivariance loss useing only affine transformation:
![](1.png)

equivariance loss useing only affine transformation + vertical and horizontal flip:
![](2.png)

![](v2.png)
