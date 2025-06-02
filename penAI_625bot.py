import discord
from discord.ext import commands
import requests
import json
import os
from dotenv import load_dotenv
import datetime

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv() 

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("DEEPSEEK_API_KEY") 
BOT_OWNER_ID_STR = os.getenv("BOT_OWNER_ID")
OPENROUTER_API_URL = os.getenv("DEEPSEEK_API_URL") 

# --- VALIDATE CRITICAL CONFIGURATION ---
if not DISCORD_BOT_TOKEN:
    print("Error: DISCORD_BOT_TOKEN was not found. Ensure it's in your .env file and the .env file is in the same folder as the script.")
    exit()
if not OPENROUTER_API_KEY:
    print("Error: DEEPSEEK_API_KEY (your OpenRouter key) was not found. Ensure it's in your .env file.")
    exit()
if not OPENROUTER_API_URL:
    print("Error: DEEPSEEK_API_URL (for OpenRouter) was not found. Ensure it's in your .env file.")
    exit()
if not BOT_OWNER_ID_STR:
    print("Error: BOT_OWNER_ID was not found. Ensure it's in your .env file.")
    exit()

try:
    BOT_OWNER_ID = int(BOT_OWNER_ID_STR)
except ValueError:
    print(f"Error: BOT_OWNER_ID '{BOT_OWNER_ID_STR}' from .env file is not a valid integer.")
    exit()

COMMAND_PREFIX = "!" 
OPENROUTER_MODEL_IDENTIFIER = "deepseek/deepseek-r1-0528-qwen3-8b:free"


# --- OPENROUTER API INTERACTION ---
conversation_history = {} 

async def get_openrouter_response(user_id: int, user_message_text: str):
    global conversation_history
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": user_message_text})
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL_IDENTIFIER, 
        "messages": conversation_history[user_id],
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        api_response_json = response.json()
        if api_response_json.get("choices") and len(api_response_json["choices"]) > 0:
            assistant_message_obj = api_response_json["choices"][0].get("message")
            if assistant_message_obj and "content" in assistant_message_obj:
                assistant_message = assistant_message_obj["content"]
                conversation_history[user_id].append({"role": "assistant", "content": assistant_message})
                return assistant_message
        print(f"Unexpected OpenRouter API response structure: {api_response_json}")
        return "Sorry, I received an unusual response from the AI."
    except requests.exceptions.Timeout:
        print("Error: Timeout calling OpenRouter API.")
        return "Sorry, the AI is taking too long to respond."
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        if e.response is not None:
            print(f"API Response Status: {e.response.status_code}, Text: {e.response.text}")
        return "Sorry, I'm having trouble connecting to the AI service."
    except Exception as e:
        print(f"Unexpected error in get_openrouter_response: {type(e).__name__} - {e}")
        return "An unexpected error occurred with the AI."

# --- DISCORD BOT SETUP ---
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, owner_id=BOT_OWNER_ID)

@bot.event
async def on_ready():
    print(f'{bot.user.name} (ID: {bot.user.id}) has connected to Discord!')
    print(f'Command Prefix: {COMMAND_PREFIX}')
    print(f'Recognized Bot Owner ID: {bot.owner_id}') 
    print(f"penAI 2-06-2025 is ready and using OpenRouter model: {OPENROUTER_MODEL_IDENTIFIER}")
    await bot.change_presence(activity=discord.Game(name=f"Chat | {COMMAND_PREFIX}help"))

def is_owner():
    async def predicate(ctx):
        if ctx.author.id != bot.owner_id: 
            await ctx.send("Sorry, only the bot owner can use this command.")
            return False
        return True
    return commands.check(predicate)

# --- ADMINISTRATIVE COMMANDS ---
@bot.command(name='ban', help='Bans a user. Usage: !ban @user [reason]')
@commands.has_permissions(ban_members=True)
@is_owner()
async def ban_member(ctx, member: discord.Member, *, reason="No reason provided."):
    if member == ctx.author or member.id == bot.owner_id:
        await ctx.send("You cannot ban yourself or the bot owner.")
        return
    if member.top_role >= ctx.guild.me.top_role:
        await ctx.send("I cannot ban this user as their role is higher than or equal to mine.")
        return
    try:
        await member.ban(reason=f"Banned by {ctx.author.name}. Reason: {reason}")
        await ctx.send(f'{member.mention} has been banned. Reason: {reason}')
    except discord.Forbidden:
        await ctx.send("I don't have permission to ban that user.")
    except discord.HTTPException as e:
        await ctx.send(f"Failed to ban user: {e}")

@bot.command(name='mute', help='Timeouts a user. Usage: !mute @user duration[s/m/h/d] [reason]')
@commands.has_permissions(moderate_members=True)
@is_owner()
async def mute_member(ctx, member: discord.Member, duration_str: str, *, reason="No reason provided."):
    if member == ctx.author or member.id == bot.owner_id:
        await ctx.send("You cannot mute yourself or the bot owner.")
        return
    if member.top_role >= ctx.guild.me.top_role:
        await ctx.send("I cannot mute this user as their role is higher than or equal to mine.")
        return
    unit = duration_str[-1].lower()
    try: 
        value = int(duration_str[:-1])
    except ValueError:
        await ctx.send("Invalid duration value (e.g., 10m, 1h).")
        return
    if unit == 's': 
        delta = datetime.timedelta(seconds=value)
    elif unit == 'm': 
        delta = datetime.timedelta(minutes=value)
    elif unit == 'h': 
        delta = datetime.timedelta(hours=value)
    elif unit == 'd': 
        delta = datetime.timedelta(days=value)
    else:
        await ctx.send("Invalid duration unit (s, m, h, or d).")
        return
    if delta.total_seconds() <= 0 or delta.total_seconds() > 28 * 24 * 60 * 60: # Max 28 days
        await ctx.send("Duration must be positive and not exceed 28 days.")
        return
    try:
        await member.timeout(delta, reason=f"Muted by {ctx.author.name}. Reason: {reason}")
        await ctx.send(f'{member.mention} has been timed out for {duration_str}.')
    except discord.Forbidden:
        await ctx.send("I don't have permission to timeout that user.")
    except discord.HTTPException as e:
        await ctx.send(f"Failed to timeout user: {e}")

@bot.command(name='unmute', help='Removes timeout. Usage: !unmute @user')
@commands.has_permissions(moderate_members=True)
@is_owner()
async def unmute_member(ctx, member: discord.Member):
    try:
        await member.timeout(None, reason=f"Timeout removed by {ctx.author.name}.") 
        await ctx.send(f'{member.mention} has been unmuted.')
    except discord.Forbidden:
        await ctx.send("I don't have permission to modify that user's timeout.")
    except discord.HTTPException as e:
        await ctx.send(f"Failed to unmute user: {e}")

@bot.command(name='createchannel', help='Creates a text channel. Usage: !createchannel channel-name [Category Name or ID]')
@commands.has_permissions(manage_channels=True)
@is_owner()
async def create_channel_command(ctx, channel_name: str, *, category_input: str = None):
    guild = ctx.guild
    target_category = None
    if category_input:
        try: 
            target_category = discord.utils.get(guild.categories, id=int(category_input))
        except ValueError: 
            target_category = discord.utils.find(lambda c: c.name.lower() == category_input.lower(), guild.categories)
        if not target_category:
            await ctx.send(f"Category '{category_input}' not found. Creating channel without category.")
    try:
        safe_channel_name = channel_name.lower().replace(" ", "-")
        new_channel = await guild.create_text_channel(name=safe_channel_name, category=target_category)
        await ctx.send(f'Channel <#{new_channel.id}> created' + (f' in category "{target_category.name}"' if target_category else '') + '!')
    except discord.Forbidden:
        await ctx.send("I don't have permission to create channels" + ( " in that category." if target_category else "."))
    except discord.HTTPException as e:
        await ctx.send(f"Failed to create channel: {e}")

# --- CHAT FUNCTIONALITY (ON_MESSAGE EVENT) ---
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user: 
        return
    if message.guild is None: 
        return
    if message.content.startswith(COMMAND_PREFIX):
        await bot.process_commands(message)
        return
        
    if bot.user.mentioned_in(message) or not message.content.startswith(COMMAND_PREFIX):
        async with message.channel.typing():
            actual_message_text = message.content
            for mention in message.mentions:
                actual_message_text = actual_message_text.replace(mention.mention, '').strip()
            
            if not actual_message_text and bot.user.mentioned_in(message): # If message was only a mention
                await message.reply("Yes? How can I help?", mention_author=False)
                return

            if actual_message_text: # Ensure there's actual text to send to the AI
                response_text = await get_openrouter_response(message.author.id, actual_message_text)
                if response_text:
                    if len(response_text) > 2000:
                        await message.reply(response_text[:2000] + "\n*[Message truncated]*", mention_author=False)
                    else:
                        await message.reply(response_text, mention_author=False)

# --- ERROR HANDLING for commands ---
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound): 
        return 
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing: `{error.param.name}`. Usage: `{COMMAND_PREFIX}help {ctx.command.name}`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Invalid argument. Usage: `{COMMAND_PREFIX}help {ctx.command.name}`.")
    elif isinstance(error, commands.UserInputError):
        await ctx.send(f"Input error for `{ctx.command.name}`: {error}")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission for this.")
    elif isinstance(error, commands.BotMissingPermissions):
        await ctx.send(f"I'm missing permissions: `{'`, `'.join(error.missing_permissions)}`")
    elif isinstance(error, commands.CheckFailure): 
        pass 
    elif isinstance(error, commands.CommandInvokeError):
        original = error.original
        print(f"Error in command {ctx.command}: {type(original).__name__} - {original}")
        await ctx.send(f"Error executing command: {type(original).__name__}")
    else:
        print(f'Unhandled error for {ctx.command}: {type(error).__name__} - {error}')
        await ctx.send("An unexpected error occurred.")

# --- RUN THE BOT ---
if __name__ == "__main__":
    print(f"Starting penAI 2-06-2025 with model '{OPENROUTER_MODEL_IDENTIFIER}'...")
    bot.run(DISCORD_BOT_TOKEN) # Uses your Discord Bot Token from .env