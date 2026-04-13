import urllib.request, zipfile, os, time

os.makedirs('meme_images', exist_ok=True)

images = [
    ("01_longcat.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/eaf107ddf133293247a30ceb6db6a815ac69604b.jpg"),
    ("02_woman_yelling_cat.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/7f8d4db9e0950e87be4c8bac37d3ac6a6338ae4e.jpg"),
    ("03_distracted_bf.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/88538518a1933c8c9119565e45ea2e4ea6a633d4.jpg"),
    ("04_this_is_fine.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/86767786e44e07dcc722111e873990d6fc0c41ab.jpg"),
    ("05_surprised_pikachu.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/464dbbbf70cad65b1d82f7e7dff1f5622fcb53a5.jpg"),
    ("06_doge.jpg", "https://upload.wikimedia.org/wikipedia/commons/d/df/Doge_homemade_meme.jpg"),
    ("07_drake_meme.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/9db55fd85a7c3db93a342696573d0cf4beee0da5.jpg"),
    ("08_grumpy_cat.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Grumpy_Cat_%2814556024763%29_%28cropped%29.jpg/960px-Grumpy_Cat_%2814556024763%29_%28cropped%29.jpg"),
    ("09_stonks.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/72f48be071f97bfe2cad05d7e674bdf939a82c64.jpg"),
    ("10_hide_pain_harold.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/2cd86354f128a9d2e2fb164d1c271b286d07aa92.jpg"),
    ("11_blinking_guy.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/f8b7a9056f13431a0f483a4a983e2b707625b7df.jpg"),
    ("12_expanding_brain.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/568e4630051d24d65d07ee957adebddf7099c9ed.jpg"),
    ("13_success_kid.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/e1008a03b9c2dadcb7069d33f237dcda6ff25434.jpg"),
    ("14_fry_not_sure.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/2fe6b0a67ffc2e4fe2bfa732e2ba9fe26f792ef3.jpg"),
    ("15_galaxy_brain.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/ccfe6f95de8d58ff4367741282543fdab56ce0fb.jpg"),
    ("16_spongebob_uno.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/0eb1e65e463653bdf30116868342e450e6bea19d.jpg"),
    ("17_longcat_collage.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/5a2d2d320cc8a4ac77d534bdd2d28558178875ad.jpg"),
    ("18_confused_stonks.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/86f92ad5e073833d1cb09b6a8b78092ad43fcf97.jpg"),
    ("19_woman_cat_ukiyoe.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/86b3ad71091ad071d9343b2a88ff151451cf2029.jpg"),
    ("20_anime_hand_shoulder.jpg", "https://pplx-res.cloudinary.com/image/upload/pplx_search_images/5a2d2d320cc8a4ac77d534bdd2d28558178875ad.jpg"),
]

headers = {'User-Agent': 'Mozilla/5.0'}

for fname, url in images:
    fpath = f'meme_images/{fname}'
    print(f'Downloading {fname}...')
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r, open(fpath, 'wb') as f:
            f.write(r.read())
        time.sleep(0.3)
    except Exception as e:
        print(f'  FAILED: {e}')

with zipfile.ZipFile('meme_images.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname, _ in images:
        fpath = f'meme_images/{fname}'
        if os.path.exists(fpath):
            zf.write(fpath, fname)

print('Done! meme_images.zip created.')