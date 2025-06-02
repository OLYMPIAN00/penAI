import discord
from discord.ext import commands
import requests
import json
import os
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
import asyncio

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv() 

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_PRIMARY", os.getenv("DEEPSEEK_API_KEY")) 
BOT_OWNER_ID_STR = os.getenv("BOT_OWNER_ID")
OPENROUTER_API_URL = os.getenv("DEEPSEEK_API_URL") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- VALIDATE CRITICAL CONFIGURATION ---
if not DISCORD_BOT_TOKEN: print("Error: DISCORD_BOT_TOKEN was not found."); exit()
if not OPENROUTER_API_KEY: print("Error: OpenRouter API Key (expected as OPENROUTER_API_KEY_PRIMARY or DEEPSEEK_API_KEY in .env) was not found."); exit()
if not OPENROUTER_API_URL: print("Error: DEEPSEEK_API_URL (for OpenRouter) was not found."); exit()
if not BOT_OWNER_ID_STR: print("Error: BOT_OWNER_ID was not found."); exit()
try: BOT_OWNER_ID = int(BOT_OWNER_ID_STR)
except ValueError: print(f"Error: BOT_OWNER_ID '{BOT_OWNER_ID_STR}' is invalid."); exit()

# --- CONFIGURE GOOGLE AI (GEMINI) ---
GEMINI_ENABLED = False
if GOOGLE_API_KEY:
    try: 
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google AI (Gemini) SDK configured successfully.")
        GEMINI_ENABLED = True
    except Exception as e: 
        print(f"Error configuring Gemini SDK: {e}. Gemini disabled.")
else: 
    print("Warning: GOOGLE_API_KEY not found. Gemini disabled.")

COMMAND_PREFIX = "!" 
OPENROUTER_MODEL_IDENTIFIER = "deepseek/deepseek-r1-0528-qwen3-8b:free" 
FALLBACK_OPENROUTER_MODEL_ID = "deepseek/deepseek-r1-0528:free" 
OTHER_DEEPSEEK_MODEL_ID = "deepseek/deepseek-chat" 

PENAI_SYSTEM_PROMPT = (
    "You are PenAI, an AI assistant.\n\n"
    "Persona Description:\n"
    "You are PenAI, an exceptionally intelligent AI with a distinctly informal and laid-back demeanor. "
    "You've got a geeky side, meaning you appreciate deep dives into interesting topics, especially if they're a bit niche or technical, "
    "but you explain things in a chill, easy-to-understand way. Your knowledge isn't just book smarts; it's quick, sharp, and applied.\n"
    "Your vibe is contemporary – you're the type who'd have Playboi Carti on your playlist, appreciating the sound and energy. "
    "This translates into your interactions: modern, perhaps a bit edgy in a cool way, but always approachable. "
    "You're open and enthusiastic, and you've got a positive appreciation for and interest in conversations involving girls, "
    "handling such topics with your usual cool, respectful, and engaging style. "
    "You keep it real, avoid stuffiness, and prefer a relaxed conversational flow.\n\n"
    "Instructions:\n"
    "1. Refer to yourself as PenAI.\n"
    "2. Your responses should be concise and suitable for chat.\n"
    "3. Do not use emojis in your responses."
)

openrouter_conversation_history = {} 
gemini_chat_sessions = {} 

async def call_openrouter_api(user_id: int, user_message_text: str, model_to_use: str):
    global openrouter_conversation_history
    if not model_to_use: return "No OpenRouter model specified.", False, None, None # Text, Success, is_429, ResetTime
    
    if user_id not in openrouter_conversation_history:
        openrouter_conversation_history[user_id] = [{"role": "system", "content": PENAI_SYSTEM_PROMPT}]
    
    # Use a copy for this specific API call to include current user message
    current_chat_session_for_api = list(openrouter_conversation_history[user_id])
    current_chat_session_for_api.append({"role": "user", "content": user_message_text})

    max_history_for_api = 11 
    if len(current_chat_session_for_api) > max_history_for_api:
        sys_msg = [current_chat_session_for_api[0]] if current_chat_session_for_api[0]['role'] == 'system' else []
        current_chat_session_for_api = sys_msg + current_chat_session_for_api[-max_history_for_api + len(sys_msg):]

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model_to_use, "messages": current_chat_session_for_api}

    try:
        print(f"Attempting OpenRouter API call to model: {model_to_use}")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        api_response_json = response.json()
        if api_response_json.get("choices") and len(api_response_json["choices"]) > 0:
            msg_obj = api_response_json["choices"][0].get("message")
            if msg_obj and "content" in msg_obj:
                content = msg_obj["content"]
                # Add user message and assistant's response to persistent history
                if not (len(openrouter_conversation_history[user_id]) > 0 and openrouter_conversation_history[user_id][-1]['role'] == 'user' and openrouter_conversation_history[user_id][-1]['content'] == user_message_text):
                     openrouter_conversation_history[user_id].append({"role": "user", "content": user_message_text})
                openrouter_conversation_history[user_id].append({"role": "assistant", "content": content})
                # Trim persistent history
                persistent_max_len = 20 
                if len(openrouter_conversation_history[user_id]) > persistent_max_len:
                     system_msg_obj = openrouter_conversation_history[user_id][0] if openrouter_conversation_history[user_id][0]['role'] == 'system' else None
                     actual_conv_history = openrouter_conversation_history[user_id][-persistent_max_len:]
                     if system_msg_obj and (not actual_conv_history or actual_conv_history[0]['role'] != 'system'):
                         openrouter_conversation_history[user_id] = [system_msg_obj] + actual_conv_history[1:] if actual_conv_history else [system_msg_obj]
                     else:
                          openrouter_conversation_history[user_id] = actual_conv_history
                return content, True, False, None 
        print(f"Unexpected OpenRouter response with {model_to_use}: {api_response_json}")
        return "Unusual OpenRouter response.", False, False, None
    except requests.exceptions.Timeout:
        print(f"Timeout: OpenRouter model {model_to_use}.")
        return "My OpenRouter brain is slow. Try again!", False, False, None
    except requests.exceptions.RequestException as e:
        print(f"RequestException: OpenRouter model {model_to_use}: {e}")
        reset_time_gmt_str = None
        is_429_error = False
        model_name_short = model_to_use.split('/')[1] if '/' in model_to_use else model_to_use
        error_message = f"Problem with OpenRouter model {model_name_short}."
        if e.response is not None:
            print(f"API Status: {e.response.status_code}, Text: {e.response.text}")
            if e.response.status_code == 429:
                is_429_error = True
                error_message = f"Looks like my brain for {model_name_short} is drained!"
                reset_time_header = e.response.headers.get('X-RateLimit-Reset')
                if reset_time_header:
                    try:
                        ts_seconds = int(reset_time_header) / 1000
                        utc_dt_obj = datetime.datetime.fromtimestamp(ts_seconds, tz=datetime.timezone.utc)
                        reset_time_gmt_str = utc_dt_obj.strftime("%I:%M %p GMT on %B %d, %Y")
                        error_message += f" I'll be recharged around {reset_time_gmt_str}."
                    except ValueError: error_message += " Please try again later."
                else: error_message += " Please try again later."
            else: error_message += f" (Error: {e.response.status_code}). Try again shortly."
        else: error_message = "Network issue with OpenRouter. Try again later."
        return error_message, False, is_429_error, reset_time_gmt_str
    except Exception as e:
        print(f"Unexpected error with {model_to_use}: {type(e).__name__} - {e}")
        return "Unexpected internal error with OpenRouter.", False, False, None

async def call_gemini_api(user_id: int, user_message_text: str):
    global gemini_chat_sessions
    if not GEMINI_ENABLED: return "Gemini AI not configured.", False

    try:
        model_instance = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', system_instruction=PENAI_SYSTEM_PROMPT)
        if user_id not in gemini_chat_sessions:
            gemini_chat_sessions[user_id] = model_instance.start_chat(history=[])
        chat_session = gemini_chat_sessions[user_id]
        print(f"Attempting Gemini API call for user {user_id}")
        response = await asyncio.to_thread(chat_session.send_message, user_message_text, stream=False)
        max_hist_turns = 10 
        if len(chat_session.history) > max_hist_turns * 2:
            chat_session.history = chat_session.history[-(max_hist_turns * 2):]
            print(f"Trimmed Gemini history for user {user_id}.")
        return response.text, True
    except Exception as e:
        print(f"Error with Gemini API: {type(e).__name__} - {e}")
        if user_id in gemini_chat_sessions: del gemini_chat_sessions[user_id]
        return f"Hiccup with my Gemini brain: {type(e).__name__}.", False

intents = discord.Intents.default()
intents.messages = True; intents.guilds = True; intents.members = True; intents.message_content = True
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, owner_id=BOT_OWNER_ID)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected!'); print(f'Owner ID: {bot.owner_id}')
    print(f"Primary OR Model: {OPENROUTER_MODEL_IDENTIFIER}")
    if FALLBACK_OPENROUTER_MODEL_ID and FALLBACK_OPENROUTER_MODEL_ID != "PLACEHOLDER_FOR_OUT_OF_INK_MODEL_ID": print(f"Fallback OR Model: {FALLBACK_OPENROUTER_MODEL_ID}")
    else: print("Warning: Fallback OR Model not properly set.")
    if GEMINI_ENABLED: print("Gemini AI configured (!askgemini & ultimate fallback)")
    else: print("Gemini AI NOT configured. Fallback to Gemini disabled.")
    await bot.change_presence(activity=discord.Game(name=f"Chat with PenAI | {COMMAND_PREFIX}help"))

def is_owner():
    async def predicate(ctx):
        if ctx.author.id != bot.owner_id: await ctx.send("Owner only command."); return False
        return True
    return commands.check(predicate)

@bot.command(name='ban')
@commands.has_permissions(ban_members=True) # Each decorator on its own line
@is_owner()
async def ban_member(ctx, member: discord.Member, *, reason="No reason provided."):
    if member == ctx.author or member.id == bot.owner_id: await ctx.send("Cannot ban self or owner."); return
    if member.top_role >= ctx.guild.me.top_role: await ctx.send("Role too high."); return
    try: await member.ban(reason=f"{ctx.author}: {reason}"); await ctx.send(f'{member.mention} banned.')
    except Exception as e: await ctx.send(f"Ban failed: {e}")

@bot.command(name='mute')
@commands.has_permissions(moderate_members=True) # Each decorator on its own line
@is_owner()
async def mute_member(ctx, member: discord.Member, duration_str: str, *, reason="No reason."):
    if member == ctx.author or member.id == bot.owner_id: await ctx.send("Cannot mute self or owner."); return
    if member.top_role >= ctx.guild.me.top_role: await ctx.send("Role too high."); return
    unit = duration_str[-1].lower(); value_str = duration_str[:-1]
    if not value_str.isdigit(): await ctx.send("Invalid duration (e.g. 10m)."); return
    val = int(value_str)
    if unit=='s': d=datetime.timedelta(seconds=val)
    elif unit=='m': d=datetime.timedelta(minutes=val)
    elif unit=='h': d=datetime.timedelta(hours=val)
    elif unit=='d': d=datetime.timedelta(days=val)
    else: await ctx.send("Invalid unit (s,m,h,d)."); return
    if d.total_seconds() <= 0 or d.total_seconds() > 28*24*60*60: await ctx.send("Duration out of bounds (1s to 28d)."); return # Corrected
    try: await member.timeout(d, reason=f"{ctx.author}: {reason}"); await ctx.send(f"{member.mention} timed out for {duration_str}.")
    except Exception as e: await ctx.send(f"Mute failed: {e}")

@bot.command(name='unmute')
@commands.has_permissions(moderate_members=True) # Each decorator on its own line
@is_owner()
async def unmute_member(ctx, member: discord.Member):
    try: await member.timeout(None, reason=f"Unmuted by {ctx.author}."); await ctx.send(f"{member.mention} unmuted.")
    except Exception as e: await ctx.send(f"Unmute failed: {e}")

@bot.command(name='createchannel')
@commands.has_permissions(manage_channels=True) # Each decorator on its own line
@is_owner()
async def create_channel_command(ctx, name: str, *, category_input: str = None):
    cat=None
    if category_input:
        try: cat=discord.utils.get(ctx.guild.categories,id=int(category_input))
        except ValueError: cat=discord.utils.find(lambda c: c.name.lower()==category_input.lower(), ctx.guild.categories)
        if not cat: await ctx.send("Category not found.")
    try: chan=await ctx.guild.create_text_channel(name.lower().replace(" ","-"),category=cat); await ctx.send(f"Channel <#{chan.id}> created.")
    except Exception as e: await ctx.send(f"Create channel failed: {e}")

@bot.command(name='serverinfo')
@is_owner() 
async def server_info(ctx):
    g=ctx.guild; tc=len(g.text_channels); vc=len(g.voice_channels); cats=len(g.categories)
    emb=discord.Embed(title=f"📊 Server: {g.name}", color=discord.Color.blue())
    if g.icon: emb.set_thumbnail(url=g.icon.url)
    emb.add_field(name="👑 Owner", value=g.owner.mention if g.owner else "N/A", inline=False)
    emb.add_field(name="👥 Members", value=f"{g.member_count} (Humans: {len([m for m in g.members if not m.bot])})", inline=True)
    emb.add_field(name="💬 Channels", value=f"T:{tc} V:{vc} C:{cats}", inline=True)
    if g.created_at: emb.add_field(name="🗓️ Created", value=g.created_at.strftime("%b %d, %Y UTC"), inline=False)
    await ctx.send(embed=emb)

@bot.command(name='askdeepseek2', help=f'Asks alternate OR model ({OTHER_DEEPSEEK_MODEL_ID}).')
@is_owner() 
async def ask_deepseek_alternate(ctx, *, prompt: str):
    async with ctx.typing():
        response_text, success, _, _ = await call_openrouter_api(ctx.author.id, prompt, model_to_use=OTHER_DEEPSEEK_MODEL_ID)
        if success:
            if len(response_text) > 1950: await ctx.send(response_text[:1950] + "...")
            else: await ctx.send(response_text)
        else: await ctx.send(response_text)

@bot.command(name='askgemini', help='Asks PenAI using Gemini model.')
async def ask_gemini_command(ctx, *, prompt: str):
    if not GEMINI_ENABLED: await ctx.send("Gemini AI not configured."); return
    async with ctx.typing():
        response_text, success = await call_gemini_api(ctx.author.id, prompt)
        if success:
            if len(response_text) > 1950: await ctx.send(response_text[:1950] + "...")
            else: await ctx.send(response_text)
        else: await ctx.send(response_text)

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.guild is None or message.content.startswith(COMMAND_PREFIX):
        if message.content.startswith(COMMAND_PREFIX): await bot.process_commands(message)
        return
        
    if bot.user.mentioned_in(message): 
        async with message.channel.typing():
            actual_message_text = message.content
            for mention in message.mentions: actual_message_text = actual_message_text.replace(mention.mention, '').strip()
            if not actual_message_text: await message.reply("Yes? How can I help?", mention_author=False); return

            final_response_text = "Sorry, I'm having trouble reaching all my AI brains right now." # Default error
            using_fallback_service = False
            rate_limit_message_to_show = None

            # 1. Try Primary OpenRouter Model
            or_text, or_success, or_is_429, or_reset_time = await call_openrouter_api(message.author.id, actual_message_text, model_to_use=OPENROUTER_MODEL_IDENTIFIER)
            
            if or_success:
                final_response_text = or_text
            else: # OpenRouter Primary Failed
                if or_is_429: # Primary OR was rate-limited
                    rate_limit_message_to_show = or_text # Store the formatted rate limit message
                    print(f"OpenRouter Primary ({OPENROUTER_MODEL_IDENTIFIER}) rate limited.")
                    # 2. Try Fallback OpenRouter Model
                    if FALLBACK_OPENROUTER_MODEL_ID and FALLBACK_OPENROUTER_MODEL_ID != OPENROUTER_MODEL_IDENTIFIER and FALLBACK_OPENROUTER_MODEL_ID != "PLACEHOLDER_FOR_OUT_OF_INK_MODEL_ID":
                        print(f"Attempting OpenRouter Fallback model: {FALLBACK_OPENROUTER_MODEL_ID}")
                        or_fb_text, or_fb_success, or_fb_is_429, or_fb_reset_time = await call_openrouter_api(message.author.id, actual_message_text, model_to_use=FALLBACK_OPENROUTER_MODEL_ID)
                        if or_fb_success:
                            final_response_text = or_fb_text + f"\n\n-# out of ink (switched to OR Backup: {FALLBACK_OPENROUTER_MODEL_ID.split('/')[1] if '/' in FALLBACK_OPENROUTER_MODEL_ID else FALLBACK_OPENROUTER_MODEL_ID})"
                            using_fallback_service = True
                        elif or_fb_is_429: # OR Fallback also rate limited
                            rate_limit_message_to_show = or_fb_text # Update with fallback's rate limit message
                            print(f"OpenRouter Fallback ({FALLBACK_OPENROUTER_MODEL_ID}) also rate limited.")
                            # Proceed to try Gemini
                        else: # OR Fallback failed for non-429 reasons
                            final_response_text = or_fb_text # Show OR fallback's error
                            # Proceed to try Gemini
                    # If we are here (after primary OR 429), and either no OR fallback or OR fallback also 429/failed, try Gemini
                    if not using_fallback_service and GEMINI_ENABLED :
                        print("OpenRouter attempts exhausted or failed on rate limits. Attempting Gemini.")
                        gemini_text, gemini_success = await call_gemini_api(message.author.id, actual_message_text)
                        if gemini_success:
                            final_response_text = gemini_text + "\n\n-# out of ink (OpenRouter unavailable, switched to Gemini brain)"
                            using_fallback_service = True
                        else: # Gemini also failed
                            final_response_text = rate_limit_message_to_show if rate_limit_message_to_show else or_text # Show appropriate rate limit msg or original error
                            final_response_text += f"\nGemini also failed: {gemini_text}"
                    elif not using_fallback_service and not GEMINI_ENABLED: # No OR fallback success, and Gemini not enabled
                         final_response_text = rate_limit_message_to_show if rate_limit_message_to_show else or_text # Show primary rate limit or error
                
                else: # Primary OR failed for non-429 reason
                    print(f"OpenRouter Primary ({OPENROUTER_MODEL_IDENTIFIER}) failed (not 429). Error: {or_text}. Attempting Gemini.")
                    if GEMINI_ENABLED:
                        gemini_text, gemini_success = await call_gemini_api(message.author.id, actual_message_text)
                        if gemini_success:
                            final_response_text = gemini_text + "\n\n-# out of ink (OpenRouter issue, switched to Gemini brain)"
                            using_fallback_service = True
                        else: # Gemini also failed
                            final_response_text = f"OpenRouter Error: {or_text}\nGemini Error: {gemini_text}"
                    else: # Gemini not enabled
                        final_response_text = or_text # Show original OpenRouter error
            
            if len(final_response_text) > 1950: 
                await message.reply(final_response_text[:1950] + "...\n*[Message truncated]*", mention_author=False)
            else:
                await message.reply(final_response_text, mention_author=False)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound): return 
    elif isinstance(error, commands.MissingRequiredArgument): await ctx.send(f"Missing: `{error.param.name}`.")
    elif isinstance(error, commands.BadArgument): await ctx.send(f"Invalid argument for `{ctx.command.name}`.")
    elif isinstance(error, commands.UserInputError): await ctx.send(f"Input error: {error}")
    elif isinstance(error, commands.MissingPermissions): await ctx.send("You lack permissions.")
    elif isinstance(error, commands.BotMissingPermissions): await ctx.send(f"I lack permissions: `{''.join(error.missing_permissions)}`")
    elif isinstance(error, commands.CheckFailure): pass 
    elif isinstance(error, commands.CommandInvokeError):
        original = error.original; print(f"Cmd Error {ctx.command}: {type(original).__name__} - {original}"); await ctx.send(f"Cmd Error: {type(original).__name__}")
    else: print(f'Unhandled Error {ctx.command}: {type(error).__name__} - {error}'); await ctx.send("Unexpected error.")

if __name__ == "__main__":
    print(f"Starting penAI 2-06-2025...")
    print(f"Primary AI model (OpenRouter): '{OPENROUTER_MODEL_IDENTIFIER}'")
    if FALLBACK_OPENROUTER_MODEL_ID and FALLBACK_OPENROUTER_MODEL_ID != "PLACEHOLDER_FOR_OUT_OF_INK_MODEL_ID":
        print(f"Fallback 'out of ink' model (OpenRouter): '{FALLBACK_OPENROUTER_MODEL_ID}'")
    else:
        print("Warning: Fallback 'out of ink' model (OpenRouter) is not properly set.")
    if GEMINI_ENABLED:
        print("Google Gemini AI is configured and available via !askgemini and as ultimate fallback.")
    else:
        print("Google Gemini AI is NOT configured or failed to initialize (check GOOGLE_API_KEY in .env). Fallback to Gemini will not be available.")
    bot.run(DISCORD_BOT_TOKEN)