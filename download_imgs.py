from google_images_search import GoogleImagesSearch
import requests


if __name__ == "__main__":
    gis = GoogleImagesSearch('_', '_')
    gis.search(search_params={
        'q': 'landscape',
        'num': 20,
        'imgSize': 'large',
    })

    for i, image in enumerate(gis.results()):
        response = requests.get(image.url)
        with open(f'images/image_{i}.png', 'wb') as f:
            f.write(response.content)
