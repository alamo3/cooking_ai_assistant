"""Seed data: three recipes that exercise oven, stovetop and air fryer, plus a pantry."""
from __future__ import annotations

from typing import Any, Dict, List

SEED_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "r001",
        "title": "Roast Chicken Thighs",
        "servings": 4,
        "ingredients": [
            {"id": "r001-i1", "name": "chicken thighs, bone-in", "amount": 8, "unit": None},
            {"id": "r001-i2", "name": "butter", "amount": 2, "unit": "tbsp"},
            {"id": "r001-i3", "name": "kosher salt", "amount": 1.5, "unit": "tsp"},
            {"id": "r001-i4", "name": "black pepper", "amount": 0.5, "unit": "tsp"},
            {"id": "r001-i5", "name": "garlic cloves, smashed", "amount": 4, "unit": None},
            {"id": "r001-i6", "name": "lemon, halved", "amount": 1, "unit": None},
            {"id": "r001-i7", "name": "thyme sprigs", "amount": 4, "unit": None},
        ],
        "steps": [
            {"id": "r001-s1", "text": "Preheat the oven to 425°F.", "duration_s": 900,
             "appliance": "oven", "temp_f": 425, "ingredient_ids": []},
            {"id": "r001-s2", "text": "Pat the chicken thighs dry and season all over with salt and pepper.",
             "duration_s": 300, "ingredient_ids": ["r001-i1", "r001-i3", "r001-i4"]},
            {"id": "r001-s3", "text": "Melt the butter in an oven-safe skillet over medium-high heat. Sear the thighs skin side down until deep golden, about 5 minutes.",
             "duration_s": 300, "appliance": "stovetop:1", "ingredient_ids": ["r001-i2", "r001-i1"]},
            {"id": "r001-s4", "text": "Flip the thighs. Tuck the garlic, thyme and lemon halves around them.",
             "duration_s": 60, "ingredient_ids": ["r001-i5", "r001-i6", "r001-i7"]},
            {"id": "r001-s5", "text": "Transfer the skillet to the oven and roast until the thighs reach 165°F, about 25 minutes.",
             "duration_s": 1500, "appliance": "oven", "temp_f": 425, "ingredient_ids": []},
            {"id": "r001-s6", "text": "Rest the chicken 10 minutes, loosely tented with foil.",
             "duration_s": 600, "ingredient_ids": []},
            {"id": "r001-s7", "text": "Squeeze the roasted lemon over the chicken, spoon over the pan juices and serve.",
             "duration_s": 120, "ingredient_ids": []},
        ],
    },
    {
        "id": "r002",
        "title": "Jasmine Rice",
        "servings": 4,
        "ingredients": [
            {"id": "r002-i1", "name": "jasmine rice", "amount": 1.5, "unit": "cup"},
            {"id": "r002-i2", "name": "water", "amount": 2.25, "unit": "cup"},
            {"id": "r002-i3", "name": "kosher salt", "amount": 0.5, "unit": "tsp"},
        ],
        "steps": [
            {"id": "r002-s1", "text": "Rinse the rice in a sieve until the water runs mostly clear.",
             "duration_s": 120, "ingredient_ids": ["r002-i1"]},
            {"id": "r002-s2", "text": "Combine the rice, water and salt in a pot. Bring to a boil over high heat.",
             "duration_s": 300, "appliance": "stovetop:2", "ingredient_ids": ["r002-i1", "r002-i2", "r002-i3"]},
            {"id": "r002-s3", "text": "Cover, reduce to the lowest heat and simmer 15 minutes. Don't lift the lid.",
             "duration_s": 900, "appliance": "stovetop:2", "ingredient_ids": []},
            {"id": "r002-s4", "text": "Take it off the heat and let it sit covered 10 minutes, then fluff with a fork.",
             "duration_s": 600, "ingredient_ids": []},
        ],
    },
    {
        "id": "r003",
        "title": "Air Fryer Brussels Sprouts",
        "servings": 4,
        "ingredients": [
            {"id": "r003-i1", "name": "brussels sprouts", "amount": 450, "unit": "g"},
            {"id": "r003-i2", "name": "olive oil", "amount": 1, "unit": "tbsp"},
            {"id": "r003-i3", "name": "kosher salt", "amount": 0.5, "unit": "tsp"},
            {"id": "r003-i4", "name": "black pepper", "amount": 0.25, "unit": "tsp"},
            {"id": "r003-i5", "name": "balsamic vinegar", "amount": 1, "unit": "tbsp"},
        ],
        "steps": [
            {"id": "r003-s1", "text": "Trim and halve the sprouts. Toss with the olive oil, salt and pepper.",
             "duration_s": 300, "ingredient_ids": ["r003-i1", "r003-i2", "r003-i3", "r003-i4"]},
            {"id": "r003-s2", "text": "Air fry at 375°F for 12 minutes, shaking the basket halfway through.",
             "duration_s": 720, "appliance": "air_fryer", "temp_f": 375, "ingredient_ids": []},
            {"id": "r003-s3", "text": "Toss with the balsamic vinegar and serve.",
             "duration_s": 60, "ingredient_ids": ["r003-i5"]},
        ],
    },
]

SEED_INVENTORY: List[Dict[str, Any]] = [
    {"name": "butter", "amount": 250, "unit": "g"},
    {"name": "olive oil", "amount": 500, "unit": "ml"},
    {"name": "garlic", "amount": 6, "unit": "clove"},
    {"name": "kosher salt", "amount": 500, "unit": "g"},
    {"name": "black pepper", "amount": 50, "unit": "g"},
    {"name": "jasmine rice", "amount": 1000, "unit": "g"},
    {"name": "chicken thighs", "amount": 8, "unit": None},
    {"name": "brussels sprouts", "amount": 450, "unit": "g"},
    {"name": "lemon", "amount": 2, "unit": None},
    {"name": "thyme", "amount": 1, "unit": "bunch"},
    {"name": "balsamic vinegar", "amount": 200, "unit": "ml"},
]
