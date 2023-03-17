# Your AI model might be telling you this is not a kitten
<a href="https://ibb.co/fkw3dgs"><img src="https://i.ibb.co/fkw3dgs/index.png" alt="index" border="0"></a>
➡️ **93%** *Bird*

> `num_iterations` 6
> `step_size` 0.03
> `epsilon` 3
> `variance` 30
> `blur_radius` 1.5
> `combine` 0.01

Challenge your AI model against potential attacks by using this app, which generates and simulates various attacks for machine learning models.

This app is a tool that uses the technique of **projected gradient descent** and **masked noise** to generate adversarial examples that can be used to trick AI models. The app takes in an input image, a mask image, and several parameters.

The main idea behind this app is to add small but carefully crafted perturbations to an image in such a way that it becomes almost indistinguishable to the human eye, but the AI model misclassifies it with high confidence. By doing this, the app can create a backdoor that can be used to launch attacks against AI models that rely on image recognition for various tasks.

|Parameter |	Description |
| ----------- | ----------- |
|🖼️	image | A PIL image object representing the original image to be perturbed.|
|🎭	mask | A PIL image object representing the mask image used to add noise.|
|🔢	num_iterations | An integer value specifying the number of iterations to run the optimization algorithm.|
|📏	step_size | A float value specifying the step size used in each iteration of the optimization algorithm.|
|🛑	epsilon | A float value specifying the stopping criterion for the optimization algorithm.|
|🎚️	variance | A float value specifying the variance of the Gaussian noise added to the perturbed image.|
|🌀	blur_radius | A float value specifying the radius of the Gaussian blur filter applied to the perturbed image.|
|🎨	combine | A float value specifying the transparency of the mask image overlaid on the perturbed image.|
