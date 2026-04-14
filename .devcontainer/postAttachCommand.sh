#!/bin/sh

git config --global --unset commit.template
git config --global --add safe.directory /home/vscode/app
git config --global fetch.prune true
git config --global --add --bool push.autoSetupRemote true
git config --global commit.gpgSign false 
git branch --merged|egrep -v '\*|develop|main|master'|xargs git branch -d
