module Main where

import Support

main = interact $ show . size . read
