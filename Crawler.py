from icrawler.builtin import BingImageCrawler
import os

def crawl_images(keywords, max_images, download_folder):
    for keyword in keywords:
        print(f"Starting crawling for {keyword}")

        target_dir = os.path.join(download_folder, keyword)
        
        # Switched to BingImageCrawler
        bing_crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            storage={'root_dir': target_dir}
        )

        bing_crawler.crawl(keyword=keyword, max_num=max_images)

keywords = ['casual','formal','chic','sporty']
img_nbr = 500
download = './data'

if __name__ == "__main__":
    crawl_images(keywords, img_nbr, download)