mkdir -p ./data
cd ./data

echo "----- Downloading datasets -----"

flicker2k = "Flickr2K.tar"
echo "1 / 3: $flicker2k"
if [ ! -f "$flicker2k" ]; then
    wget "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar" --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
else
    echo "$flicker2k is already downloaded."
fi

div2k_train = "DIV2K_train_HR.zip"
echo "2 / 3: $div2k_train"
if [ ! -f "$div2k_train" ]; then
    wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
else
    echo "$div2k_train is already downloaded."
fi

div2k_valid = "DIV2K_valid_HR.zip"
echo "3 / 3: $div2k_valid"
if [ ! -f "$div2k_valid" ]; then
    wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
else
    echo "$div2k_valid is already downloaded."
fi

echo "----- Unzipping datasets -----"

echo "1 / 3: Flickr2K"
tar -xf Flickr2K.tar
mv ./Flickr2K/Flickr2K_HR ./train/Flickr2K
rm -rf ./Flickr2K

echo "2 / 3: DIV2K Train"
unzip -q DIV2K_train_HR.zip -d ./train
mv ./train/DIV2K_train_HR ./train/DIV2K

echo "3 / 3: DIV2K Valid"
unzip -q DIV2K_valid_HR.zip -d ./valid
mv ./valid/DIV2K_valid_HR ./valid/DIV2K

# echo "----- Cleaning up -----"
# rm -rf ./*.zip
# rm -rf ./*.tar
