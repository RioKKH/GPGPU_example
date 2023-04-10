#!/bin/bash

nsys profile \
	-t osrt,cuda,nvtx,cublas,cudnn \
       	-o baseline \
	-w true \
	$1 
