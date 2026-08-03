#!/usr/bin/env python3
"""
CulinaryVLM — Generate all 20 canonical recipes offline.

This script creates high-quality canonical recipe JSON files for
all 20 biryani categories without any API calls. These serve as
the alignment reference for Stage 7 (multimodal alignment).

Usage:
    python pipeline/00b_generate_fallback_recipes.py

Output: configs/canonical_recipes/{category}.json (20 files)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fallback_recipes")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "configs" / "canonical_recipes"

# ═══════════════════════════════════════════════════════════════
# Common ingredient aliases (reused across recipes)
# ═══════════════════════════════════════════════════════════════

COMMON_ALIASES = {
    "chicken": {"hi": "मुर्गा/चिकन", "te": "కోడి", "ml": "കോഴി", "bn": "মুরগি", "ur": "مرغ"},
    "basmati rice": {"hi": "बासमती चावल", "te": "బాస్మతి బియ్యం", "ml": "ബസ്മതി അരി", "bn": "বাসমতী চাল", "ur": "باسمتی چاول"},
    "yogurt": {"hi": "दही", "te": "పెరుగు", "ml": "തൈര്", "bn": "দই", "ur": "دہی"},
    "onion": {"hi": "प्याज", "te": "ఉల్లిపాయ", "ml": "ഉള്ളി", "bn": "পেঁয়াজ", "ur": "پیاز"},
    "ghee": {"hi": "घी", "te": "నెయ్యి", "ml": "നെയ്യ്", "bn": "ঘি", "ur": "گھی"},
    "saffron": {"hi": "केसर", "te": "కుంకుమపువ్వు", "ml": "കുങ്കുമപ്പൂവ്", "bn": "জাফরান", "ur": "زعفران"},
    "mint": {"hi": "पुदीना", "te": "పుదీనా", "ml": "പുതിന", "bn": "পুদিনা", "ur": "پودینہ"},
    "coriander": {"hi": "धनिया", "te": "కొత్తిమీర", "ml": "മല്ലി", "bn": "ধনে", "ur": "دھنیا"},
    "green chili": {"hi": "हरी मिर्च", "te": "పచ్చి మిర్చి", "ml": "പച്ചമുളക്", "bn": "কাঁচা লঙ্কা", "ur": "ہری مرچ"},
    "ginger-garlic paste": {"hi": "अदरक-लहसुन पेस्ट", "te": "అల్లం-వెల్లుల్లి పేస్ట్", "ml": "ഇഞ്ചി-വെളുത്തുള്ളി പേസ്റ്റ്", "bn": "আদা-রসুন বাটা", "ur": "ادرک لہسن پیسٹ"},
    "oil": {"hi": "तेल", "te": "నూనె", "ml": "എണ്ണ", "bn": "তেল", "ur": "تیل"},
    "salt": {"hi": "नमक", "te": "ఉప్పు", "ml": "ഉപ്പ്", "bn": "নুন", "ur": "نمک"},
    "turmeric": {"hi": "हल्दी", "te": "పసుపు", "ml": "മഞ്ഞൾ", "bn": "হলুদ", "ur": "ہلدی"},
    "red chili powder": {"hi": "लाल मिर्च पाउडर", "te": "ఎర్ర మిర్చి పొడి", "ml": "മുളകുപൊടി", "bn": "লাল লঙ্কার গুঁড়ো", "ur": "لال مرچ پاؤڈر"},
    "cumin": {"hi": "जीरा", "te": "జీలకర్ర", "ml": "ജീരകം", "bn": "জিরা", "ur": "زیرہ"},
    "bay leaf": {"hi": "तेज पत्ता", "te": "బిర్యానీ ఆకు", "ml": "കറിവേപ്പില", "bn": "তেজপাতা", "ur": "تیز پتہ"},
    "cardamom": {"hi": "इलायची", "te": "ఏలకులు", "ml": "ഏലക്ക", "bn": "এলাচ", "ur": "الائچی"},
    "cloves": {"hi": "लौंग", "te": "లవంగాలు", "ml": "കരയാമ്പൂ", "bn": "লবঙ্গ", "ur": "لونگ"},
    "cinnamon": {"hi": "दालचीनी", "te": "దాల్చిన చెక్క", "ml": "കറുവാപ്പട്ട", "bn": "দারুচিনি", "ur": "دار چینی"},
    "lemon": {"hi": "नींबू", "te": "నిమ్మకాయ", "ml": "ചെറുനാരങ്ങ", "bn": "লেবু", "ur": "لیموں"},
    "potato": {"hi": "आलू", "te": "బంగాళాదుంప", "ml": "ഉരുളക്കിഴങ്ങ്", "bn": "আলু", "ur": "آلو"},
    "egg": {"hi": "अंडा", "te": "గుడ్డు", "ml": "മുട്ട", "bn": "ডিম", "ur": "انڈا"},
    "coconut milk": {"hi": "नारियल का दूध", "te": "కొబ్బరి పాలు", "ml": "തേങ്ങാപ്പാൽ", "bn": "নারকেলের দুধ", "ur": "ناریل کا دودھ"},
    "curry leaves": {"hi": "कढ़ी पत्ता", "te": "కరివేపాకు", "ml": "കറിവേപ്പില", "bn": "কারি পাতা", "ur": "کری پتہ"},
    "mustard oil": {"hi": "सरसों का तेल", "te": "ఆవ నూనె", "ml": "കടുക് എണ്ണ", "bn": "সরষের তেল", "ur": "سرسوں کا تیل"},
    "rose water": {"hi": "गुलाब जल", "te": "గులాబీ నీరు", "ml": "പനിനീർ", "bn": "গোলাপ জল", "ur": "عرق گلاب"},
    "kewra water": {"hi": "केवड़ा जल", "te": "కేవ్రా నీరు", "ml": None, "bn": "কেওড়া জল", "ur": "کیوڑا"},
    "star anise": {"hi": "चक्र फूल", "te": "అనాసపువ్వు", "ml": "തക്കോലം", "bn": "স্টার অ্যানিস", "ur": "بادیان"},
    "fennel seeds": {"hi": "सौंफ", "te": "సోంపు", "ml": "പെരുംജീരകം", "bn": "মৌরি", "ur": "سونف"},
    "black pepper": {"hi": "काली मिर्च", "te": "మిరియాలు", "ml": "കുരുമുളക്", "bn": "গোল মরিচ", "ur": "کالی مرچ"},
}


def pick_aliases(keys: list[str]) -> dict:
    """Pick a subset of common aliases by key."""
    return {k: COMMON_ALIASES[k] for k in keys if k in COMMON_ALIASES}


# ═══════════════════════════════════════════════════════════════
# All 20 canonical recipes
# ═══════════════════════════════════════════════════════════════

ALL_RECIPES = {

    # ──────────────────────────────────────────────────────────
    "Hyderabadi": {
        "category": "Hyderabadi",
        "full_name": "Hyderabadi Chicken Biryani",
        "region": "Telangana / Andhra Pradesh (Hyderabad)",
        "distinguishing_features": [
            "Kacchi (raw) method — raw marinated meat layered with parboiled rice",
            "Sealed dum cooking with dough-sealed lid (purdah)",
            "Generous use of saffron, fried onions (birista), and mint",
            "Basmati rice soaked and parboiled with whole spices",
            "Served with mirchi ka salan and raita"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken with yogurt, ginger-garlic paste, green chili paste, red chili powder, turmeric, biryani masala, fried onions, mint, coriander, lemon juice, salt for 2-6 hours.", "duration_minutes": 180, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "green chilies", "mint", "coriander", "fried onions", "biryani masala"]},
            {"step_number": 2, "action": "Soak basmati rice", "description": "Wash and soak long-grain basmati rice in water for 30-45 minutes.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": ["basmati rice"]},
            {"step_number": 3, "action": "Parboil rice", "description": "Boil rice with whole spices (bay leaf, cardamom, cloves, cinnamon, star anise) until 70% cooked. Drain.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cloves", "cinnamon", "star anise", "salt"]},
            {"step_number": 4, "action": "Fry onions for birista", "description": "Thinly slice onions and deep fry until dark golden brown and crispy.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["onion", "oil"]},
            {"step_number": 5, "action": "Layer marinated chicken", "description": "Spread raw marinated chicken evenly at the bottom of a heavy handi.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["chicken"]},
            {"step_number": 6, "action": "Layer parboiled rice", "description": "Spread parboiled rice evenly over the chicken layer.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["basmati rice"]},
            {"step_number": 7, "action": "Add saffron and garnish", "description": "Drizzle saffron milk, ghee, fried onions, mint, coriander over the rice.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "ghee", "onion", "mint", "coriander"]},
            {"step_number": 8, "action": "Seal with dough (purdah)", "description": "Seal the lid with wheat flour dough to trap all steam.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 9, "action": "Cook on dum", "description": "High heat 5 min, then very low heat 25-35 min. Do not open.", "duration_minutes": 35, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 10, "action": "Rest and serve", "description": "Rest 10 min, break seal, gently mix layers. Serve with raita and mirchi ka salan.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "mint", "coriander", "green chili", "ginger-garlic paste"]),
        "key_spices": ["saffron", "cardamom", "cloves", "cinnamon", "star anise", "mace", "nutmeg", "bay leaf"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy-bottomed handi",
        "dum_method": "Sealed dum with dough (purdah)",
        "typical_duration_minutes": 120,
        "notes": "Uses kacchi (raw) method. Nizam's kitchen legacy. Always served with mirchi ka salan."
    },

    # ──────────────────────────────────────────────────────────
    "Kolkata": {
        "category": "Kolkata",
        "full_name": "Kolkata Chicken Biryani",
        "region": "West Bengal (Kolkata)",
        "distinguishing_features": [
            "Light, subtly flavored biryani with less spice than Hyderabadi",
            "Signature use of whole boiled potatoes and eggs alongside chicken",
            "Meat is pre-cooked (pakki method), not raw layered",
            "Rose water and kewra water as aromatic finishers",
            "Nutmeg and mace dominant over chili heat"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken with yogurt, ginger-garlic paste, mild spices (nutmeg, mace, cardamom), salt for 1-2 hours.", "duration_minutes": 90, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste"]},
            {"step_number": 2, "action": "Cook chicken (pakki method)", "description": "Cook marinated chicken in ghee with onions until 80% done. Reserve gravy.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["chicken", "ghee", "onion"]},
            {"step_number": 3, "action": "Boil potatoes and eggs", "description": "Boil whole potatoes and eggs separately. Lightly fry potatoes until golden.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["potato", "egg", "oil"]},
            {"step_number": 4, "action": "Soak and parboil rice", "description": "Soak basmati 30 min, parboil with whole spices until 70% done.", "duration_minutes": 40, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 5, "action": "Layer chicken", "description": "Place cooked chicken with gravy at bottom of pot.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["chicken"]},
            {"step_number": 6, "action": "Add potatoes and eggs", "description": "Place fried potatoes and halved boiled eggs around the chicken.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["potato", "egg"]},
            {"step_number": 7, "action": "Layer rice", "description": "Spread parboiled rice over the chicken and potato layer.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["basmati rice"]},
            {"step_number": 8, "action": "Add aromatics", "description": "Drizzle saffron milk, rose water, kewra water, ghee, and fried onions over rice.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "rose water", "kewra water", "ghee", "onion"]},
            {"step_number": 9, "action": "Dum cooking", "description": "Seal pot, cook on low heat for 20-25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 10, "action": "Rest and serve", "description": "Rest 10 min, gently mix. Serve with change'r dal and raita.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "potato", "egg", "rose water", "kewra water"]),
        "key_spices": ["nutmeg", "mace", "cardamom", "cinnamon", "bay leaf", "saffron"],
        "rice_type": "Long-grain Basmati (Gobindobhog for some variants)",
        "cooking_vessel": "Heavy-bottomed degchi",
        "dum_method": "Sealed dum (cloth or dough)",
        "typical_duration_minutes": 100,
        "notes": "Adapted from Awadhi/Lucknowi style by Wajid Ali Shah's cooks after exile to Kolkata. Potatoes were added as a cost-saving measure."
    },

    # ──────────────────────────────────────────────────────────
    "Lucknowi": {
        "category": "Lucknowi",
        "full_name": "Lucknowi (Awadhi) Chicken Biryani",
        "region": "Uttar Pradesh (Lucknow / Awadh)",
        "distinguishing_features": [
            "Pakki (cooked) method — meat is fully cooked before layering",
            "Very aromatic: heavy use of ittar, rose water, kewra water",
            "Subtle spicing — less heat, more fragrance than Hyderabadi",
            "Rice is cooked separately in spiced yakhni (stock)",
            "Potli masala (spice bundle) technique"
        ],
        "steps": [
            {"step_number": 1, "action": "Prepare yakhni", "description": "Simmer chicken bones/pieces with whole spices, onion, ginger to make aromatic stock.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "ginger-garlic paste", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 2, "action": "Marinate chicken", "description": "Marinate chicken in yogurt, mild spices, salt for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt"]},
            {"step_number": 3, "action": "Cook chicken fully (pakki)", "description": "Cook marinated chicken in ghee with fried onions until completely tender.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "ghee", "onion"]},
            {"step_number": 4, "action": "Cook rice in yakhni", "description": "Cook basmati rice in the prepared yakhni stock until 75% done.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 5, "action": "Prepare potli masala", "description": "Tie whole spices in muslin cloth to form a spice bundle. Place in pot.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["cardamom", "cloves", "cinnamon", "star anise"]},
            {"step_number": 6, "action": "Layer chicken and rice", "description": "Alternate layers of cooked chicken and yakhni rice.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 7, "action": "Add aromatics", "description": "Drizzle saffron milk, rose water, kewra water, and ghee.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "rose water", "kewra water", "ghee"]},
            {"step_number": 8, "action": "Dum cooking", "description": "Seal and cook on very low heat for 20 minutes.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 9, "action": "Rest and serve", "description": "Rest 10 min, remove potli masala, gently mix. Serve with korma or raita.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "rose water", "kewra water", "cardamom", "cinnamon"]),
        "key_spices": ["saffron", "cardamom", "mace", "nutmeg", "kewra", "rose water", "bay leaf"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy degchi with sealed lid",
        "dum_method": "Sealed dum (gentle, low heat)",
        "typical_duration_minutes": 110,
        "notes": "Nawabi cuisine — emphasis on fragrance over heat. Potli masala is removed before serving."
    },

    # ──────────────────────────────────────────────────────────
    "Malabar": {
        "category": "Malabar",
        "full_name": "Malabar/Thalassery Chicken Biryani",
        "region": "Kerala (Malabar coast / Thalassery)",
        "distinguishing_features": [
            "Uses Kaima/Jeerakasala rice instead of Basmati",
            "Coconut milk in gravy and sometimes in rice",
            "Distinct Kerala spice profile: fennel, star anise, Malabar pepper",
            "Meat cooked in thick masala gravy before layering",
            "Fried cashews and raisins as garnish"
        ],
        "steps": [
            {"step_number": 1, "action": "Prepare masala paste", "description": "Grind coconut, ginger, garlic, green chilies, fennel, poppy seeds into thick paste.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["ginger-garlic paste", "green chili", "fennel seeds", "coconut milk"]},
            {"step_number": 2, "action": "Cook chicken in masala", "description": "Cook chicken pieces in the masala paste with onions, tomatoes, curry leaves, coconut milk until tender.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "curry leaves", "coconut milk"]},
            {"step_number": 3, "action": "Wash and soak Kaima rice", "description": "Wash Kaima/Jeerakasala rice and soak for 20 minutes.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 4, "action": "Cook rice", "description": "Cook rice with whole spices until 80% done. Drain.", "duration_minutes": 12, "is_defining_step": False, "ingredients_used": ["basmati rice", "cardamom", "cloves", "cinnamon"]},
            {"step_number": 5, "action": "Fry onions", "description": "Fry thinly sliced onions in coconut oil until crispy golden.", "duration_minutes": 12, "is_defining_step": False, "ingredients_used": ["onion", "oil"]},
            {"step_number": 6, "action": "Fry cashews and raisins", "description": "Fry cashew nuts and raisins in ghee until golden.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["ghee"]},
            {"step_number": 7, "action": "Layer chicken and rice", "description": "Alternate layers of chicken masala and partially cooked rice.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 8, "action": "Add garnish and aromatics", "description": "Top with fried onions, cashews, raisins, ghee, saffron milk, and mint.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["ghee", "saffron", "mint", "onion"]},
            {"step_number": 9, "action": "Dum cooking", "description": "Seal and cook on low heat for 20-25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 10, "action": "Serve", "description": "Gently fluff, serve with raita and pickle.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "coconut milk", "onion", "ghee", "curry leaves", "fennel seeds", "green chili", "ginger-garlic paste", "mint"]),
        "key_spices": ["fennel", "star anise", "Malabar pepper", "cardamom", "cinnamon", "cloves", "curry leaves"],
        "rice_type": "Kaima / Jeerakasala (short-grain)",
        "cooking_vessel": "Heavy uruli or cheena chatti",
        "dum_method": "Sealed dum with banana leaf or lid",
        "typical_duration_minutes": 100,
        "notes": "Thalassery biryani is distinct from other Indian biryanis due to Kaima rice and Mappila Muslim culinary traditions."
    },

    # ──────────────────────────────────────────────────────────
    "Sindhi": {
        "category": "Sindhi",
        "full_name": "Sindhi/Karachi Chicken Biryani",
        "region": "Sindh / Karachi (Pakistani origin, Indian Sindhi diaspora)",
        "distinguishing_features": [
            "Heavy use of dried plums (aloo bukhara) and potatoes",
            "Yogurt-based marinade with generous red chili and cumin",
            "Fried potatoes layered with rice",
            "More heavily spiced than Lucknowi, less sweet than Kolkata",
            "Often uses tomatoes in the chicken masala"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken in yogurt, ginger-garlic paste, red chili, turmeric, cumin, salt for 1-2 hours.", "duration_minutes": 90, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "red chili powder", "cumin", "turmeric"]},
            {"step_number": 2, "action": "Cook chicken", "description": "Cook marinated chicken with onions, tomatoes, green chilies until tender.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": ["chicken", "onion", "green chili"]},
            {"step_number": 3, "action": "Prepare dried plums", "description": "Soak dried plums (aloo bukhara) in warm water.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 4, "action": "Fry potatoes", "description": "Peel and halve potatoes, fry in oil until golden on all sides.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["potato", "oil"]},
            {"step_number": 5, "action": "Parboil rice", "description": "Cook basmati rice with whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cumin"]},
            {"step_number": 6, "action": "Layer chicken", "description": "Place cooked chicken at the bottom, add fried potatoes and soaked plums.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["chicken", "potato"]},
            {"step_number": 7, "action": "Layer rice", "description": "Add rice layer with saffron milk, fried onions, mint, coriander.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["basmati rice", "saffron", "onion", "mint", "coriander"]},
            {"step_number": 8, "action": "Dum cooking", "description": "Seal pot with dough or foil, cook low heat 25-30 min.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 9, "action": "Serve", "description": "Mix gently, serve with raita and salad.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "potato", "cumin", "red chili powder", "green chili", "mint", "saffron"]),
        "key_spices": ["cumin", "red chili", "turmeric", "cardamom", "bay leaf", "black pepper"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy-bottomed pot (deg)",
        "dum_method": "Sealed dum",
        "typical_duration_minutes": 110,
        "notes": "Karachi-style biryani is known for its spice intensity. Dried plums add a unique tangy-sweet note."
    },

    # ──────────────────────────────────────────────────────────
    "Muradabadi": {
        "category": "Muradabadi",
        "full_name": "Muradabadi Chicken Biryani",
        "region": "Uttar Pradesh (Moradabad)",
        "distinguishing_features": [
            "Uses mustard oil for a pungent, distinctive flavor",
            "Yakhni-based — rice cooked in meat stock",
            "Less oil/ghee than Hyderabadi or Lucknowi",
            "Minimal use of fried onions compared to other styles",
            "Simple spicing — cumin, black pepper, bay leaf dominate"
        ],
        "steps": [
            {"step_number": 1, "action": "Prepare yakhni", "description": "Boil chicken pieces with onions, whole spices, salt to make yakhni stock. Remove chicken, strain stock.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "bay leaf", "cardamom", "black pepper"]},
            {"step_number": 2, "action": "Soak rice", "description": "Wash and soak basmati rice for 30 minutes.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": ["basmati rice"]},
            {"step_number": 3, "action": "Cook chicken in mustard oil", "description": "Fry boiled chicken pieces in mustard oil with cumin, green chilies until browned.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["chicken", "mustard oil", "cumin", "green chili"]},
            {"step_number": 4, "action": "Cook rice in yakhni", "description": "Cook soaked rice in the yakhni stock until 80% done.", "duration_minutes": 12, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 5, "action": "Layer chicken and rice", "description": "Place fried chicken at bottom, layer yakhni rice on top.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Add saffron and mint", "description": "Drizzle saffron-infused mustard oil and fresh mint leaves.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "mustard oil", "mint"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal and cook on very low heat for 20 minutes.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Fluff gently, serve with green chutney and raita.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "mustard oil", "onion", "cumin", "black pepper", "bay leaf", "saffron", "mint", "green chili"]),
        "key_spices": ["cumin", "black pepper", "bay leaf", "cardamom", "cinnamon"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy degchi",
        "dum_method": "Sealed dum (low heat)",
        "typical_duration_minutes": 90,
        "notes": "Known for mustard oil usage and yakhni-based cooking. Less elaborate than Lucknowi but deeply flavored."
    },

    # ──────────────────────────────────────────────────────────
    "Delhi": {
        "category": "Delhi",
        "full_name": "Delhi/Old Delhi Chicken Biryani",
        "region": "Delhi (Old Delhi / Purani Dilli)",
        "distinguishing_features": [
            "Mughal-influenced, heavily spiced with robust flavors",
            "Generous use of ghee and fried onions (birista)",
            "Meat cooked in thick, rich gravy before layering",
            "Often includes yogurt, tomatoes, and fried cashews",
            "Street food heritage — served in large degs at weddings"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate in yogurt, ginger-garlic paste, red chili, garam masala, salt for 2 hours.", "duration_minutes": 120, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "red chili powder"]},
            {"step_number": 2, "action": "Fry onions", "description": "Fry large quantity of sliced onions in ghee until dark golden for birista.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["onion", "ghee"]},
            {"step_number": 3, "action": "Cook chicken in masala", "description": "Cook marinated chicken with fried onions, tomatoes in ghee until rich and thick.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "ghee"]},
            {"step_number": 4, "action": "Parboil rice", "description": "Cook basmati with whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon", "cloves"]},
            {"step_number": 5, "action": "Layer chicken and rice", "description": "Alternate layers of chicken masala and rice in a large deg.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Add saffron and ghee", "description": "Drizzle saffron milk, melted ghee, fried onions, mint on top.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "ghee", "onion", "mint"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal with dough or heavy lid, cook on low heat 30 minutes.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Mix gently, serve with raita, salad, and shorba.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "mint", "ginger-garlic paste", "red chili powder", "coriander"]),
        "key_spices": ["garam masala", "cardamom", "cinnamon", "cloves", "bay leaf", "mace", "saffron"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Large deg (cauldron)",
        "dum_method": "Sealed dum with dough",
        "typical_duration_minutes": 120,
        "notes": "Old Delhi biryani has Mughal roots, popularized at Jama Masjid area restaurants."
    },

    # ──────────────────────────────────────────────────────────
    "Andhra": {
        "category": "Andhra",
        "full_name": "Andhra Chicken Biryani",
        "region": "Andhra Pradesh / Telangana (various sub-regions)",
        "distinguishing_features": [
            "Spiciest biryani style — heavy use of green and red chilies",
            "Uses Seeraga Samba or regular Basmati rice",
            "Coconut and curry leaves in the masala",
            "Generous use of ginger and garlic",
            "Each sub-region has variants: Vizag, Nellore, Guntur, Rayalaseema"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, red chili powder, turmeric, ginger-garlic paste, green chilies, salt for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "red chili powder", "turmeric", "ginger-garlic paste", "green chili"]},
            {"step_number": 2, "action": "Cook chicken masala", "description": "Cook chicken in oil with onions, tomatoes, curry leaves, coconut paste until gravy thickens.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "oil", "curry leaves"]},
            {"step_number": 3, "action": "Parboil rice", "description": "Cook basmati with bay leaf, cardamom, cloves until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cloves"]},
            {"step_number": 4, "action": "Prepare green chili paste", "description": "Grind green chilies with mint and coriander for the spicy layer.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["green chili", "mint", "coriander"]},
            {"step_number": 5, "action": "Layer chicken and rice", "description": "Layer chicken masala, green chili paste, and rice alternately.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Add aromatics", "description": "Add saffron milk, ghee, fried onions, curry leaves on top.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "ghee", "onion", "curry leaves"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal and cook on low heat for 25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Mix gently, serve with mirchi ka salan or onion raita.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "curry leaves", "green chili", "red chili powder", "turmeric", "ginger-garlic paste"]),
        "key_spices": ["red chili", "green chili", "turmeric", "cumin", "curry leaves", "cardamom", "cinnamon"],
        "rice_type": "Basmati or Seeraga Samba",
        "cooking_vessel": "Heavy-bottomed pot",
        "dum_method": "Sealed dum",
        "typical_duration_minutes": 100,
        "notes": "Known as the spiciest Indian biryani. Sub-regional variants: Vizag (seafood-influenced), Guntur (extra chili), Rayalaseema (rustic)."
    },

    # ──────────────────────────────────────────────────────────
    "Ambur": {
        "category": "Ambur",
        "full_name": "Ambur Chicken Biryani",
        "region": "Tamil Nadu (Ambur, Vellore district)",
        "distinguishing_features": [
            "Uses Seeraga Samba rice (jeera rice) — short-grain, very aromatic",
            "Dahi-based (curd-based) marination, no tomato",
            "Relatively less spiced — lets rice and meat flavor dominate",
            "Mint and coriander used sparingly",
            "Traditionally served with dalcha (lentil-vegetable curry) and brinjal gravy"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken with yogurt, ginger-garlic paste, chili powder, turmeric, mint for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "red chili powder", "mint"]},
            {"step_number": 2, "action": "Cook chicken", "description": "Cook marinated chicken in oil with onions until tender.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": ["chicken", "onion", "oil"]},
            {"step_number": 3, "action": "Wash Seeraga Samba rice", "description": "Wash the short-grain Seeraga Samba rice thoroughly.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 4, "action": "Cook rice in meat stock", "description": "Cook rice in the chicken cooking liquid (stock) with whole spices.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 5, "action": "Layer and dum", "description": "Layer chicken and rice, add ghee, fried onions. Seal and dum for 15 min.", "duration_minutes": 15, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice", "ghee", "onion"]},
            {"step_number": 6, "action": "Serve", "description": "Serve with dalcha and brinjal gravy (traditional sides).", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "mint", "ginger-garlic paste", "red chili powder", "bay leaf", "cardamom"]),
        "key_spices": ["cardamom", "cinnamon", "bay leaf", "cumin", "fennel"],
        "rice_type": "Seeraga Samba (short-grain jeera rice)",
        "cooking_vessel": "Earthen pot or heavy pan",
        "dum_method": "Short dum (15 min)",
        "typical_duration_minutes": 80,
        "notes": "One of Tamil Nadu's most famous biryanis. Seeraga Samba rice absorbs flavors beautifully. Always served with dalcha."
    },

    # ──────────────────────────────────────────────────────────
    "Dindigul": {
        "category": "Dindigul",
        "full_name": "Dindigul Chicken Biryani",
        "region": "Tamil Nadu (Dindigul)",
        "distinguishing_features": [
            "Uses Seeraga Samba rice like Ambur but spicier",
            "Cube-cut chicken pieces (smaller than other styles)",
            "Generous use of pepper — peppery heat over chili heat",
            "Curd-marinated, no tomato",
            "Drier texture compared to other biryanis"
        ],
        "steps": [
            {"step_number": 1, "action": "Cut chicken into small cubes", "description": "Cut chicken into small, uniform cube-shaped pieces (defining characteristic).", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["chicken"]},
            {"step_number": 2, "action": "Marinate chicken", "description": "Marinate cubed chicken with yogurt, pepper, ginger-garlic paste, turmeric for 1 hour.", "duration_minutes": 60, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "black pepper", "ginger-garlic paste", "turmeric"]},
            {"step_number": 3, "action": "Cook chicken masala", "description": "Cook chicken with onions, green chilies, curry leaves in oil until dry and well-cooked.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "green chili", "curry leaves", "oil"]},
            {"step_number": 4, "action": "Cook Seeraga Samba rice", "description": "Cook rice with whole spices until fully done.", "duration_minutes": 15, "is_defining_step": False, "ingredients_used": ["basmati rice", "cardamom", "cinnamon", "bay leaf"]},
            {"step_number": 5, "action": "Mix chicken and rice", "description": "Mix the cooked chicken masala with rice (toss method, not strict layering).", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Serve", "description": "Serve hot with raita and brinjal gravy.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "black pepper", "curry leaves", "green chili", "ginger-garlic paste", "turmeric", "oil"]),
        "key_spices": ["black pepper", "cumin", "fennel", "cardamom", "curry leaves"],
        "rice_type": "Seeraga Samba (short-grain)",
        "cooking_vessel": "Heavy pan",
        "dum_method": "No traditional dum — toss/mix method",
        "typical_duration_minutes": 70,
        "notes": "Dindigul is famous for its peppery, dry-style biryani. The cube-cut chicken and toss method are unique identifiers."
    },

    # ──────────────────────────────────────────────────────────
    "Bihari": {
        "category": "Bihari",
        "full_name": "Bihari/Champaran Chicken Biryani",
        "region": "Bihar (Champaran / Patna)",
        "distinguishing_features": [
            "Cooked in a sealed matka (earthen pot) or handi",
            "Mustard oil is the primary cooking fat",
            "Simple, rustic spicing — onion, ginger-garlic, whole spices",
            "No elaborate layering — chicken and rice cooked together",
            "Smoky flavor from mustard oil and sealed cooking"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, mustard oil, ginger-garlic paste, red chili, salt for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "mustard oil", "ginger-garlic paste"]},
            {"step_number": 2, "action": "Heat mustard oil", "description": "Heat mustard oil to smoking point in a handi, let it cool slightly.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["mustard oil"]},
            {"step_number": 3, "action": "Cook onions and whole spices", "description": "Fry sliced onions with bay leaf, cardamom, cloves in mustard oil.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["onion", "mustard oil", "bay leaf", "cardamom", "cloves"]},
            {"step_number": 4, "action": "Add chicken", "description": "Add marinated chicken, cook on high heat until browned.", "duration_minutes": 15, "is_defining_step": False, "ingredients_used": ["chicken"]},
            {"step_number": 5, "action": "Add soaked rice", "description": "Add soaked basmati rice directly on top of the chicken with water.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 6, "action": "Add mint and seal", "description": "Add mint leaves, green chilies. Seal pot with dough.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["mint", "green chili"]},
            {"step_number": 7, "action": "Dum in matka", "description": "Cook on low heat for 30 minutes. Do not open.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Break seal, serve directly from the handi.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "mustard oil", "mint", "green chili", "ginger-garlic paste", "bay leaf", "cardamom"]),
        "key_spices": ["bay leaf", "cardamom", "cloves", "black pepper", "cumin"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Matka (earthen pot) or handi",
        "dum_method": "Sealed dum in matka/handi",
        "typical_duration_minutes": 90,
        "notes": "Champaran meat biryani (originally mutton) is Bihar's most famous biryani. Rustic, one-pot style."
    },

    # ──────────────────────────────────────────────────────────
    "Assamese": {
        "category": "Assamese",
        "full_name": "Assamese Chicken Biryani",
        "region": "Assam (Northeast India)",
        "distinguishing_features": [
            "Uses local Joha rice (aromatic short-grain) instead of Basmati",
            "Mustard oil as primary cooking fat",
            "Less spice than North Indian styles — subtle, clean flavors",
            "Bhut jolokia (ghost pepper) for heat in some variants",
            "Influence of Mughal cuisine adapted with local ingredients"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, ginger-garlic paste, turmeric, mustard oil, salt for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "turmeric", "mustard oil"]},
            {"step_number": 2, "action": "Cook chicken", "description": "Cook marinated chicken in mustard oil with onions until tender.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": ["chicken", "mustard oil", "onion"]},
            {"step_number": 3, "action": "Cook Joha rice", "description": "Cook local aromatic rice with whole spices until 80% done.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 4, "action": "Layer and dum", "description": "Layer chicken and rice, add ghee, mint. Seal and cook 20 min.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice", "ghee", "mint"]},
            {"step_number": 5, "action": "Serve", "description": "Serve with pickled onion and raita.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "mustard oil", "turmeric", "ginger-garlic paste", "ghee", "bay leaf", "mint"]),
        "key_spices": ["bay leaf", "cardamom", "cinnamon", "cumin", "turmeric"],
        "rice_type": "Joha rice (local aromatic) or Basmati",
        "cooking_vessel": "Heavy-bottomed pot",
        "dum_method": "Sealed dum",
        "typical_duration_minutes": 80,
        "notes": "Assamese biryani reflects Northeast Indian adaptation of Mughal cuisine. Joha rice gives a distinctive fragrance."
    },

    # ──────────────────────────────────────────────────────────
    "Bombay": {
        "category": "Bombay",
        "full_name": "Bombay/Mumbai Chicken Biryani",
        "region": "Maharashtra (Mumbai)",
        "distinguishing_features": [
            "Blend of North and South Indian influences",
            "Uses fried potatoes (like Kolkata) and fried onions (like Hyderabadi)",
            "Coconut and khus khus (poppy seeds) in the masala paste",
            "Often includes crispy fried chicken pieces",
            "Served with burhani raita (spiced yogurt drink)"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, ginger-garlic paste, spice mix, lemon juice for 1-2 hours.", "duration_minutes": 90, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "lemon"]},
            {"step_number": 2, "action": "Cook chicken masala", "description": "Cook chicken with onions, coconut paste, tomatoes in oil until thick gravy forms.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "onion", "oil"]},
            {"step_number": 3, "action": "Fry potatoes", "description": "Fry potato halves in oil until golden.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["potato", "oil"]},
            {"step_number": 4, "action": "Parboil rice", "description": "Cook basmati with whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 5, "action": "Layer everything", "description": "Layer chicken masala, fried potatoes, rice. Repeat.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "potato", "basmati rice"]},
            {"step_number": 6, "action": "Add saffron and ghee", "description": "Drizzle saffron milk, ghee, fried onions, mint on top layer.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["saffron", "ghee", "onion", "mint"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal and cook on low heat 25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Serve with burhani raita.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "potato", "ghee", "saffron", "mint", "ginger-garlic paste", "lemon"]),
        "key_spices": ["cardamom", "cinnamon", "cloves", "bay leaf", "mace", "star anise"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy-bottomed pot",
        "dum_method": "Sealed dum",
        "typical_duration_minutes": 110,
        "notes": "Mumbai biryani blends influences from multiple Indian biryani traditions. Burhani raita is the signature accompaniment."
    },

    # ──────────────────────────────────────────────────────────
    "Kashmiri": {
        "category": "Kashmiri",
        "full_name": "Kashmiri Chicken Biryani",
        "region": "Jammu & Kashmir",
        "distinguishing_features": [
            "Uses Kashmiri red chili — vibrant color, mild heat",
            "Generous use of dry fruits: almonds, cashews, raisins",
            "Saffron-heavy, fragrant with fennel and dry ginger",
            "Often includes yogurt-based gravy (rogan josh style)",
            "No tomato in traditional preparation"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, Kashmiri red chili, fennel powder, dry ginger, saffron for 2 hours.", "duration_minutes": 120, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "red chili powder", "fennel seeds", "saffron"]},
            {"step_number": 2, "action": "Cook chicken in ghee", "description": "Cook chicken in ghee with onions, whole spices until tender and rich.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "ghee", "onion", "cardamom", "cinnamon"]},
            {"step_number": 3, "action": "Toast dry fruits", "description": "Fry almonds, cashews, raisins in ghee until golden.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["ghee"]},
            {"step_number": 4, "action": "Parboil rice", "description": "Cook basmati with saffron and whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "saffron", "cardamom", "bay leaf"]},
            {"step_number": 5, "action": "Layer chicken and rice", "description": "Alternate layers of chicken and saffron rice.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Add dry fruits and saffron", "description": "Scatter toasted dry fruits, drizzle saffron milk and ghee.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "ghee"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal and cook on low heat 25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Garnish with more saffron strands and dry fruits. Serve with raita.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "fennel seeds", "red chili powder", "cardamom", "cinnamon"]),
        "key_spices": ["saffron", "Kashmiri red chili", "fennel", "dry ginger", "cardamom", "cinnamon", "bay leaf"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy copper degchi",
        "dum_method": "Sealed dum",
        "typical_duration_minutes": 110,
        "notes": "Kashmiri biryani is the mildest in heat but richest in saffron and dry fruits. Kashmiri red chili gives color without burn."
    },

    # ──────────────────────────────────────────────────────────
    "Mughlai": {
        "category": "Mughlai",
        "full_name": "Mughlai Chicken Biryani",
        "region": "North India (Mughal court cuisine)",
        "distinguishing_features": [
            "Rich, elaborate preparation — the 'royal' biryani style",
            "Cream, cashew paste, and ghee create a rich gravy",
            "Multiple marination steps (yogurt then cream)",
            "Liberal use of saffron, rose water, kewra water",
            "Whole spices dominate — minimal ground masala"
        ],
        "steps": [
            {"step_number": 1, "action": "First marination", "description": "Marinate chicken in yogurt, ginger-garlic paste, salt for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste"]},
            {"step_number": 2, "action": "Second marination", "description": "Add cream, cashew paste, saffron milk to chicken. Rest 30 min.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": ["chicken", "saffron"]},
            {"step_number": 3, "action": "Cook chicken in ghee", "description": "Cook double-marinated chicken in generous ghee with whole spices.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken", "ghee", "cardamom", "cinnamon", "cloves"]},
            {"step_number": 4, "action": "Parboil rice", "description": "Cook basmati with whole spices, saffron until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "saffron", "bay leaf"]},
            {"step_number": 5, "action": "Layer chicken and rice", "description": "Alternate rich chicken gravy and saffron rice layers.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Add royal aromatics", "description": "Drizzle rose water, kewra water, saffron milk, melted ghee.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["rose water", "kewra water", "saffron", "ghee"]},
            {"step_number": 7, "action": "Dum cooking", "description": "Seal with dough, dum on low heat 25 minutes.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": []},
            {"step_number": 8, "action": "Serve", "description": "Serve with shahi korma and raita.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "ghee", "saffron", "rose water", "kewra water", "cardamom", "cinnamon", "cloves"]),
        "key_spices": ["saffron", "cardamom", "mace", "nutmeg", "cinnamon", "cloves", "bay leaf"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Silver-lined degchi (traditionally)",
        "dum_method": "Sealed dum with dough",
        "typical_duration_minutes": 120,
        "notes": "The original court biryani of the Mughal emperors. Richest and most elaborate style."
    },

    # ──────────────────────────────────────────────────────────
    "Bamboo": {
        "category": "Bamboo",
        "full_name": "Bamboo Chicken Biryani",
        "region": "Andhra Pradesh / Tribal regions",
        "distinguishing_features": [
            "Cooked inside a fresh bamboo trunk — completely unique vessel",
            "Bamboo imparts a subtle sweet, earthy, smoky flavor",
            "Usually cooked over an open wood fire",
            "Simple marination — the bamboo is the flavor agent",
            "Tribal/rustic origin — minimal spice, maximum natural flavor"
        ],
        "steps": [
            {"step_number": 1, "action": "Prepare bamboo trunk", "description": "Select a fresh green bamboo trunk, clean inside, create an opening at one end.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 2, "action": "Marinate chicken", "description": "Marinate chicken with simple spices: turmeric, red chili, ginger-garlic paste, salt.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": ["chicken", "turmeric", "red chili powder", "ginger-garlic paste"]},
            {"step_number": 3, "action": "Stuff bamboo", "description": "Add marinated chicken, soaked rice, onions, green chilies, mint into the bamboo trunk.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["chicken", "basmati rice", "onion", "green chili", "mint"]},
            {"step_number": 4, "action": "Add water and seal", "description": "Pour measured water into bamboo, seal the opening with banana leaf or dough.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 5, "action": "Cook over fire", "description": "Place sealed bamboo over open wood fire, rotate periodically for 45-60 minutes.", "duration_minutes": 50, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 6, "action": "Serve from bamboo", "description": "Split bamboo open, scoop out biryani. The inner bamboo layer adds flavor.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "onion", "green chili", "mint", "turmeric", "red chili powder", "ginger-garlic paste", "salt", "oil"]),
        "key_spices": ["turmeric", "red chili", "cumin", "bay leaf"],
        "rice_type": "Basmati or local rice",
        "cooking_vessel": "Fresh bamboo trunk (defining feature)",
        "dum_method": "Sealed bamboo over open fire",
        "typical_duration_minutes": 90,
        "notes": "Completely unique cooking vessel. The bamboo membrane caramelizes and infuses the biryani with distinctive flavor."
    },

    # ──────────────────────────────────────────────────────────
    "Degi": {
        "category": "Degi",
        "full_name": "Degi Chicken Biryani",
        "region": "North India (wedding/celebration style)",
        "distinguishing_features": [
            "Cooked in a large deg (cauldron) — designed for feeding crowds",
            "Extra ghee and oil for rich, moist texture at scale",
            "Yakhni-based: rice cooked in meat stock",
            "Multiple layers in the large deg",
            "Wedding/dawat staple — quantity-focused recipe"
        ],
        "steps": [
            {"step_number": 1, "action": "Prepare yakhni in deg", "description": "Cook large quantity of chicken with bones in a deg with whole spices to make yakhni.", "duration_minutes": 40, "is_defining_step": True, "ingredients_used": ["chicken", "bay leaf", "cardamom", "cinnamon", "onion"]},
            {"step_number": 2, "action": "Cook chicken in masala", "description": "In the deg, cook chicken with onions, yogurt, and spices until semi-tender.", "duration_minutes": 25, "is_defining_step": False, "ingredients_used": ["chicken", "onion", "yogurt", "ghee"]},
            {"step_number": 3, "action": "Parboil rice in yakhni", "description": "Cook soaked rice in the yakhni stock until 60-70% done.", "duration_minutes": 12, "is_defining_step": True, "ingredients_used": ["basmati rice"]},
            {"step_number": 4, "action": "Layer in the deg", "description": "Arrange multiple layers of chicken and rice in the large deg.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 5, "action": "Add generous ghee", "description": "Pour melted ghee, saffron milk, fried onions, and mint between layers.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["ghee", "saffron", "onion", "mint"]},
            {"step_number": 6, "action": "Seal and dum", "description": "Seal deg with heavy lid and dough, cook on charcoal dum for 40 minutes.", "duration_minutes": 40, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 7, "action": "Serve", "description": "Open deg, serve in portions with raita and salad.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "mint", "coriander", "bay leaf", "cardamom"]),
        "key_spices": ["cardamom", "cinnamon", "cloves", "bay leaf", "mace", "nutmeg", "saffron"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Large deg (cauldron)",
        "dum_method": "Sealed dum with charcoal on top",
        "typical_duration_minutes": 150,
        "notes": "Degi biryani is the Indian wedding biryani — cooked in massive quantities in large copper degs."
    },

    # ──────────────────────────────────────────────────────────
    "Matka": {
        "category": "Matka",
        "full_name": "Matka Chicken Biryani",
        "region": "Various (method-based, not region-specific)",
        "distinguishing_features": [
            "Cooked and served in a clay pot (matka/handi)",
            "Clay pot imparts earthy flavor and retains heat",
            "Usually sealed with dough for dum effect",
            "The matka is broken or opened at the table for presentation",
            "Retains moisture exceptionally well"
        ],
        "steps": [
            {"step_number": 1, "action": "Season the matka", "description": "Soak new clay matka in water for 30 min, then grease with oil.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": ["oil"]},
            {"step_number": 2, "action": "Marinate chicken", "description": "Marinate with yogurt, spices, ginger-garlic paste for 1 hour.", "duration_minutes": 60, "is_defining_step": False, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste"]},
            {"step_number": 3, "action": "Cook chicken", "description": "Pre-cook marinated chicken with onions in a pan until 80% done.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": ["chicken", "onion", "oil"]},
            {"step_number": 4, "action": "Parboil rice", "description": "Cook rice with whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom"]},
            {"step_number": 5, "action": "Fill the matka", "description": "Layer chicken and rice inside the matka. Add saffron milk, ghee, mint.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["chicken", "basmati rice", "saffron", "ghee", "mint"]},
            {"step_number": 6, "action": "Seal matka with dough", "description": "Seal the matka opening with wheat flour dough to create airtight dum.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 7, "action": "Dum in matka", "description": "Place sealed matka on low heat or in oven for 30 minutes.", "duration_minutes": 30, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 8, "action": "Serve from matka", "description": "Break the dough seal at the table, serve directly from the clay pot.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "mint", "ginger-garlic paste", "bay leaf", "cardamom"]),
        "key_spices": ["cardamom", "cinnamon", "bay leaf", "cloves", "saffron"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Clay matka/handi (defining feature)",
        "dum_method": "Sealed dum in clay pot",
        "typical_duration_minutes": 100,
        "notes": "The clay matka is the defining feature. Earthen vessel cooking is common across many Indian cuisines."
    },

    # ──────────────────────────────────────────────────────────
    "Tandoori": {
        "category": "Tandoori",
        "full_name": "Tandoori Chicken Biryani",
        "region": "North India (method-based)",
        "distinguishing_features": [
            "Uses pre-cooked tandoori chicken instead of raw/curry chicken",
            "Smoky, charred flavor from tandoor-roasted chicken",
            "Tandoori masala (yogurt, Kashmiri chili, smoky spices) as base",
            "Chicken is pulled off the bone and mixed with biryani",
            "Combines two iconic Indian dishes: tandoori chicken + biryani"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate for tandoori", "description": "Marinate chicken in tandoori masala: yogurt, Kashmiri chili, lemon, ginger-garlic, mustard oil, garam masala.", "duration_minutes": 120, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "red chili powder", "lemon", "ginger-garlic paste", "mustard oil"]},
            {"step_number": 2, "action": "Grill tandoori chicken", "description": "Grill/roast marinated chicken in tandoor, oven, or on grill until charred and cooked.", "duration_minutes": 25, "is_defining_step": True, "ingredients_used": ["chicken"]},
            {"step_number": 3, "action": "Parboil rice", "description": "Cook basmati with whole spices until 70% done.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cinnamon"]},
            {"step_number": 4, "action": "Prepare gravy base", "description": "Make a light gravy with onions, tomatoes, green chilies, cream.", "duration_minutes": 15, "is_defining_step": False, "ingredients_used": ["onion", "green chili", "oil"]},
            {"step_number": 5, "action": "Layer tandoori chicken", "description": "Place grilled tandoori chicken pieces (bone-in or shredded) in the gravy.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["chicken"]},
            {"step_number": 6, "action": "Layer rice and dum", "description": "Add rice on top, drizzle saffron, ghee, mint. Seal and dum 20 min.", "duration_minutes": 20, "is_defining_step": False, "ingredients_used": ["basmati rice", "saffron", "ghee", "mint"]},
            {"step_number": 7, "action": "Serve", "description": "Mix gently, serve with raita and green chutney.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "onion", "ghee", "saffron", "mint", "lemon", "red chili powder", "ginger-garlic paste"]),
        "key_spices": ["Kashmiri red chili", "garam masala", "cardamom", "cumin", "coriander", "smoked paprika"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Tandoor for chicken + heavy pot for biryani",
        "dum_method": "Sealed dum after tandoori step",
        "typical_duration_minutes": 110,
        "notes": "Fusion of two iconic dishes. The smoky tandoori chicken elevates the biryani with char flavor."
    },

    # ──────────────────────────────────────────────────────────
    "Arabic": {
        "category": "Arabic",
        "full_name": "Arabic/Arabian Chicken Biryani (Mandi/Kabsa)",
        "region": "Middle East / Arabian Peninsula (adopted in India)",
        "distinguishing_features": [
            "Uses Arabian spice blend: bezar, loomi (dried lime), baharat",
            "Chicken is often smoked or slow-roasted, not curry-based",
            "Saffron rice with raisins and roasted nuts",
            "Less chili heat, more warm spice complexity",
            "Tomato-based chicken preparation (in Kabsa variant)"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate with yogurt, baharat spice mix, lemon, ginger-garlic for 2 hours.", "duration_minutes": 120, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "lemon", "ginger-garlic paste"]},
            {"step_number": 2, "action": "Roast chicken", "description": "Slow-roast or grill whole chicken pieces until golden and cooked through.", "duration_minutes": 35, "is_defining_step": True, "ingredients_used": ["chicken", "ghee"]},
            {"step_number": 3, "action": "Prepare saffron rice", "description": "Cook basmati rice with saffron, cardamom, cinnamon, dried lime (loomi) until done.", "duration_minutes": 20, "is_defining_step": True, "ingredients_used": ["basmati rice", "saffron", "cardamom", "cinnamon"]},
            {"step_number": 4, "action": "Toast nuts and raisins", "description": "Fry almonds, cashews, pine nuts, raisins in ghee.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["ghee"]},
            {"step_number": 5, "action": "Assemble", "description": "Place roasted chicken on a bed of saffron rice. Garnish with toasted nuts and raisins.", "duration_minutes": 10, "is_defining_step": True, "ingredients_used": ["chicken", "basmati rice"]},
            {"step_number": 6, "action": "Serve", "description": "Serve with dakous (tomato-chili sauce) and yogurt.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": []}
        ],
        "ingredient_aliases": pick_aliases(["chicken", "basmati rice", "yogurt", "ghee", "saffron", "cardamom", "cinnamon", "lemon", "ginger-garlic paste", "onion"]),
        "key_spices": ["baharat", "saffron", "cardamom", "cinnamon", "dried lime (loomi)", "cumin", "black pepper"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Roasting tray + large serving platter",
        "dum_method": "No dum — roast/grill + assemble",
        "typical_duration_minutes": 120,
        "notes": "Arabian biryani (Mandi/Kabsa) differs fundamentally from Indian biryani — roasted meat over flavored rice rather than layered dum."
    },
}


def main() -> None:
    """Generate all fallback recipes."""
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Generating All Canonical Recipes  ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for category, recipe in sorted(ALL_RECIPES.items()):
        filename = category.lower().replace("/", "_").replace(" ", "_") + ".json"
        filepath = OUTPUT_DIR / filename

        if filepath.exists():
            logger.info(f"  ⊘ {filename:25s} — already exists, skipping")
            skipped += 1
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=2, ensure_ascii=False)

        steps = len(recipe.get("steps", []))
        aliases = len(recipe.get("ingredient_aliases", {}))
        logger.info(f"  ✓ {filename:25s} — {steps:2d} steps, {aliases:2d} aliases")
        generated += 1

    logger.info(f"\n  Generated: {generated}, Skipped: {skipped}, Total: {len(ALL_RECIPES)}")
    logger.info("✓ All canonical recipes ready!")


if __name__ == "__main__":
    main()
