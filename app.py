import discord
from discord.ext import commands
from discord import app_commands
from PIL import Image
import onnxruntime as ort
import numpy as np
import aiohttp, io, os


TOKEN = os.getenv("DISCORD_TOKEN")

# ------------------------------
# 1️⃣ Load ONNX model
# ------------------------------
MODEL_PATH = "vit_b_32_visual.onnx"  # your downloaded ONNX visual model
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("ONNX model loaded successfully!")

# ------------------------------
# 2️⃣ Image preprocess
# ------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0  # <-- ensure float32

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img.astype(np.float32)  # ensure float32

def encode_image(img: Image.Image):
    inp = preprocess_image(img)
    feat = session.run([output_name], {input_name: inp})[0]
    feat /= np.linalg.norm(feat, axis=-1, keepdims=True)
    return feat

# ------------------------------
# 3️⃣ Load Pokémon database
# ------------------------------
pokemon_dir = "pokemon_images"
pokemon_names = []
pokemon_features = []

POKEMON_INFO_FILE = "pkdex.txt"
pokemon_info = {}

# Load info
if os.path.exists(POKEMON_INFO_FILE):
    with open(POKEMON_INFO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(" ", 1)
                if len(parts) > 1:
                    name = parts[1].split("#")[0].strip().lower()
                    pokemon_info[name] = line.strip()

# Precompute features
print("Encoding Pokémon images...")
for fname in os.listdir(pokemon_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name = os.path.splitext(fname)[0]
        img = Image.open(os.path.join(pokemon_dir, fname))
        feat = encode_image(img)
        pokemon_features.append(feat)
        pokemon_names.append(name)

pokemon_features = np.concatenate(pokemon_features, axis=0)
print(f"Loaded {len(pokemon_names)} Pokémon images.")

# ------------------------------
# 4️⃣ Discord bot setup
# ------------------------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix=None, intents=intents)

@bot.event
async def on_ready():
    bot.session = aiohttp.ClientSession()
    try:
        await bot.tree.sync()
        print("Commands synced.")
    except Exception as e:
        print("Sync error:", e)
    print(f"Logged in as {bot.user}")

# ------------------------------
# 5️⃣ Context menu: Identify Pokémon
# ------------------------------
@bot.tree.context_menu(name="Identify Pokémon")
async def identify_pokemon(interaction: discord.Interaction, message: discord.Message):
    image_url = None

    # attachments
    if message.attachments:
        att = message.attachments[0]
        if att.content_type and att.content_type.startswith("image"):
            image_url = att.url

    # embeds
    if not image_url and message.embeds:
        embed = message.embeds[0]
        if embed.image and embed.image.url:
            image_url = embed.image.url

    if not image_url:
        await interaction.response.send_message("❌ No image found!", ephemeral=True)
        return

    # download image
    async with bot.session.get(image_url) as resp:
        if resp.status != 200:
            await interaction.response.send_message("⚠️ Could not download image!", ephemeral=True)
            return
        data = await resp.read()

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        input_feat = encode_image(img)

        # cosine similarity
        similarities = input_feat @ pokemon_features.T
        best_idx = similarities.argmax()
        best_score = similarities[0, best_idx]
        best_name = pokemon_names[best_idx]

        if best_score > 0.85:
            # custom text
            text_path = os.path.join("pokemon_texts", f"{best_name}.txt")
            info_line = f"@Pokétwo#8236 c {best_name}"
            if os.path.exists(text_path):
                try:
                    with open(text_path, "r", encoding="utf-8") as f:
                        info_line = f.read().strip()
                except:
                    pass

            # send image
            img_path = os.path.join(pokemon_dir, f"{best_name}.png")
            if os.path.exists(img_path):
                file = discord.File(img_path, filename=f"{best_name}.png")
                await interaction.response.send_message(file=file, content=info_line, ephemeral=True)
        else:
            await interaction.response.send_message(
                "❓ I couldn’t confidently identify this Pokémon.",
                ephemeral=True
            )

    except Exception as e:
        await interaction.response.send_message(f"Error: {e}", ephemeral=True)

@bot.event
async def on_close():
    await bot.session.close()

bot.run(TOKEN)
