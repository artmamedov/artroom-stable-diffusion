# Welcome to Artroom!

## Artroom is a No Code Local GUI that lets you generate AI Art directly on your computer! 
Currently, Artroom only supports NVIDIA GPUs and have a recommended amount of 8GB VRAM (4GB minimum). This will be updated in the future. 

![8](https://user-images.githubusercontent.com/59179719/202012552-75ef881b-e6fa-4b3f-8b4e-191a1f3ce240.PNG)

# Support 
If you have questions, comments, or just want to stop by and chat, [join the Discord](https://discord.gg/XNEmesgTFy)

If you want to see what's coming and what's currently being worked on, [check out our roadmap](https://trello.com/b/S9hoQBFK/artroom-roadmap)

# Installation
To install, download the latest Artroom version at:

## Download
https://artroom.ai/download-app

OR

https://github.com/artmamedov/artroom-stable-diffusion/releases

Note: This is a hefty installer (~20GB total) so make sure you have room on your device and a stable internet connection. Any future update of Artroom will automatically update the app. No need to do anything else.

## Installer
When you open the installer, you'll be prompted to choose your model weights path (~17GB) and your regular installation path (~2GB)

![0](https://user-images.githubusercontent.com/59179719/202014459-abcf5a45-4472-4614-9f49-b3376ac37477.PNG)
![1](https://user-images.githubusercontent.com/59179719/202014463-35de5b5d-23cc-4cf4-8c91-9ccbaca60f95.PNG)

You will then be prompted to press OK, leading to a command prompt window opening up. This is the actual installer, so please do not close this window. It takes a while, so grab a ☕ and come back in a little bit. (Especially the pytorch part. It's not stuck, it's just taking a while to finish up).
![3](https://user-images.githubusercontent.com/59179719/202014782-2248da3d-223d-45a3-8998-91864fd3dacf.PNG)

NOTE: If you already have model weights (from AUTOMATIC or another local GUI), you can actually close the window once you get to the model weight download. This has been intentionally left as the last step so it can be interrupted without breaking any core functionality. You will have to link the folder that holds your weights in Settings (shown later).

![4](https://user-images.githubusercontent.com/59179719/202014785-8c2e74c6-c99b-407c-bff1-0f109832e6a9.PNG)

## Enjoy

Once you're done, you're ready to go!

![5](https://user-images.githubusercontent.com/59179719/202014956-e5c4f752-2dac-4735-86a9-65677a14f7f6.PNG)


# Using the App

![6](https://user-images.githubusercontent.com/59179719/202007566-6131e31f-c74d-4b8a-9026-d1876285c3d2.PNG)

## Create

The main things you will need to do is just type in your prompt and press run! Everything else will work as is. As you get more comfortable, try exploring with the different settings on the right side. If you have questions, go through the tutorial, check out the (?) bubbles, check out our docs, or just reach out on Discord. The community will be more than happy to help!

![8](https://user-images.githubusercontent.com/59179719/202016453-b0d86c74-33eb-4436-a4ac-eeb3bebec5a8.PNG)


## Paint
In the paint tab, you can Paint on your image and have that go into your drawing OR you can use the mask function so that ONLY the part you painted over gets masked. You can also do a reverse mask so that everything EXCEPT what you painted over gets used. Note that these requrie a starting image to work. You can use any color (except white) as your paint color.

KNOWN BUG: If the Paint tool is stuck, try going to a new tab and back. Will fix in later updates. 

![17](https://user-images.githubusercontent.com/59179719/202015453-47224d71-0935-4890-a3d2-2709e03aa5fa.PNG)

## Queue
When you add an image, it goes into the Queue. Here you can keep track of all of your runs and just keep putting in prompts/settings. You can switch models and it will use the model set at the time of generation. This way, you don't have to wait for one batch of images to finish before starting the next one. 

![18](https://user-images.githubusercontent.com/59179719/202016326-d596f463-f700-46af-a97c-1d84831bb6f0.PNG)

## Upscale
In the Upscale section, you can use RealESRGAN to make your image bigger or GFPGAN to fix awkawrdness in faces.
![19](https://user-images.githubusercontent.com/59179719/202015990-88546a97-9500-4147-b6a3-aee91c8c4020.PNG)


## Settings
In Settings, you can customize your experience by choosing different models, your generation speed (will take up more memory), how long to wait in between queues, and a few other tweaks. 

If you have a bug and are unsure why, you can turn on Debug Mode in Settings to have a console open as you generate. This will help with better error reporting. Some users like to have it on the entire time. 

![20](https://user-images.githubusercontent.com/59179719/202016003-39377a24-40f5-4c84-ba92-14b733057dd6.PNG)


## Help

The Artroom App has built in tutorials and documentation, as well as a direct link to the Discord. If you have any questions, feel free to reach out there or on Github or Reddit or wherever you find yourself. You can also email at artur@artroom.ai. 


![7](https://user-images.githubusercontent.com/59179719/202015184-ae83f91c-1612-4282-b275-d448e36d499f.PNG)


# Follow Us/Learn More:
- Twitter: @ArtroomAI
- Documentation: https://docs.equilibriumai.com/artroom
