import os
import nextcord
from nextcord.abc import GuildChannel
from nextcord import Interaction, SlashOption, ChannelType
from nextcord.ext import commands
from craiyon import Craiyon
from io import BytesIO
import time
from PIL import Image
# import base64
import requests
# import random as rt
import mimetypes
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

TOKEN = os.getenv('DISCORD_TOKEN')

SUPPORTED_MIMETYPES = ["image/jpeg", "image/png", "image/webp"]

bot = commands.Bot(command_prefix="!", intents=nextcord.Intents.all())

testingServerID = 937529035859316766
#All Functions Used by Bot:

# Load the waifu2x model
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")


# Define a function to upscale an image using the waifu2x model
def upscale_image(image_path, scale=2):
  # Load the image and convert it to a NumPy array
  response = requests.get(image_path)
  image = Image.open(BytesIO(response.content)).convert('RGB')
  image_array = np.array(image)

  # Rescale the image array to the desired size
  size = image_array.shape[:2]
  new_size = (size[0] * scale, size[1] * scale)
  image_array = tf.image.resize(image_array, new_size, method='bicubic')

  # Normalize the image array
  image_array = tf.cast(image_array, tf.float32) / 255.0

  # Upscale the image using the waifu2x model
  image_tensor = tf.expand_dims(image_array, axis=0)
  upscaled_tensor = model(image_tensor)

  # Convert the upscaled tensor back to a NumPy array
  upscaled_array = np.array(upscaled_tensor[0])

  # Rescale the upscaled array to the original size
  upscaled_image_array = tf.image.resize(upscaled_array,
                                         size,
                                         method='bicubic')

  # Convert the upscaled array to an image and return it
  upscaled_image = Image.fromarray(np.uint8(upscaled_image_array.numpy() *
                                            255))
  return upscaled_image


#Bot Setup:


@bot.event
async def on_ready():
  print("Twilight is ready!")


#Bot Commands:


@bot.slash_command(description="Generates AI Images")
async def dream(interaction=Interaction, prompt=str):
  ETA = int(time.time() + 60)
  msg = await interaction.send(
    content=
    f"Go grab a snack, this may take a while... ETA: <t:{ETA}:R> \n \n Remember, the ETA does not mean your image will arrive at the end of the ETA.",
    ephemeral=False)
  generator = Craiyon()
  result = generator.generate(prompt)
  images = result.images
  for image_url in images:
    count = 0
  image = requests.get(image_url).content
  imageReturn = BytesIO(image)
  await msg.delete()
  msg = await interaction.send(
    content="Image Generated (_Courtesy of craiyon.com_)",
    file=nextcord.File(imageReturn, "ImageGenerated.jpg"))


@bot.slash_command(
  description="Upscales Images. \n Supported Image Types: Webp, JPG, PNG")
async def upscale(interaction=Interaction):
  if len(interaction.message.attachments) > 0:
    attachment_url = interaction.message.attachments[0].url
    image = Image.open(requests.get(attachment_url, stream=True).raw)

  else:
    await interaction.response.send_message(
      'Please upload an image file. Run this command again with image file.')


@bot.slash_command(description="Shuts Down Bot")
@commands.is_owner()
async def shutdown(interaction=Interaction):
  print("Bot Closed")
  await bot.change_presence(status=nextcord.Status.invisible)
  msg = await interaction.response.send_message(f"Bot has shutdown.")
  await bot.close()


#Bot Run:

if __name__ == "__main__":
  bot.run(TOKEN)
