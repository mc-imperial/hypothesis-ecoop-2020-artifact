module Main where

import Support

main = interact $ show . prop_parse . read
