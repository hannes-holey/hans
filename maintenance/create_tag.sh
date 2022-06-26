#!/bin/bash

major=$(git tag --sort=-creatordate | awk -F "." 'NR==1{print $1}')
minor=$(git tag --sort=-creatordate | awk -F "." 'NR==1{print $2}')
patch=$(git tag --sort=-creatordate | awk -F "." 'NR==1{print $3}')

if [ "$#" -ne 1 ]
then
echo "Specify exactly 1 input argument (1-3)"
exit 0
fi

if [[ $1 -eq 0 ]] || [ $1 -gt 3 ]
then
echo "Input argument has to be either 1 (increment major), 2 (increment minor), or 3 (increment patch)"
exit 0
fi

if [ $1 -eq 1 ]
then
    let "major=major+1"
    minor=0
    patch=0
elif [ $1 -eq 2 ]
then
    let "minor=minor+1"
    patch=0
elif [ $1 -eq 3 ]
then
    let "patch=patch+1"
fi

version=$major.$minor.$patch
echo "Creating new tag ($version)"
git tag -a $version -m "version $version"
