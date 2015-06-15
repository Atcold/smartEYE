# Get data from server

path='https://engineering.purdue.edu/elab/smartEYE-data/'

echo ''
echo 'Do you wish to download the network?'
select yn in 'Yes' 'No'; do
    case $yn in
        Yes ) mkdir -p net
              wget $path'/net/17cate9filter.tar.gz' .
              tar -xzf 17cate9filter.tar.gz
              mv 17cate9filter net/
              rm *.tar.gz
              break;;
        No ) break;;
    esac
done
