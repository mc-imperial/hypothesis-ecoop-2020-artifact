{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

-- | Divide by 0 example in a simple arithmetic language.

module Main where

import System.Environment
import Test.QuickCheck
import Control.Monad

import GHC.Generics
import Data.Typeable
import Data.Int

import EvalCommon

#ifdef USE_SMARTCHECK
import Test.SmartCheck
#endif

newtype J = J { getInt :: Int16 } deriving (Show, Read, Typeable, Generic, Eq, Ord)

customIntShrinker :: (Int16, Int16) -> [(Int16, Int16)]
customIntShrinker (0, _) = []
customIntShrinker (_, 0) = []
customIntShrinker (m, n) | m < n = [(m + n - s, s) | s <- shrink n]
customIntShrinker _ = []


shrinkAdjacentPairs :: I -> [I]
shrinkAdjacentPairs [] = []
shrinkAdjacentPairs [x] = []
shrinkAdjacentPairs (x1@(J n1) : x2@(J n2) : xs) = (J (n1 + n2) : xs) : [(J s1) : (J s2) : xs | (s1, s2) <- customIntShrinker (n1, n2)] ++ (map (x1:) . shrinkAdjacentPairs $ x2 : xs)

reorderListShrinker :: (Ord a) => [a] -> [[a]]
reorderListShrinker [] = []
reorderListShrinker [x] = []
reorderListShrinker (x : y : rest) | x > y = (y : x : rest) : (map (x:) $ reorderListShrinker (y : rest))
reorderListShrinker (x : y : rest) = (map (x:)) $ reorderListShrinker (y : rest)


instance Arbitrary J where
  arbitrary = fmap J arbitrary
#ifndef USE_SMARTCHECK
  shrink (J n) = map J (shrink n)
#endif

type I = [J]

data T = T I I I I I deriving (Read, Show, Typeable, Generic)

#ifdef USE_SMARTCHECK
instance SubTypes J
instance SubTypes T
#endif


combine :: [[a]] -> [a]
combine ([]:xs) = combine xs
combine ([y]:xs) = y : combine xs
combine ((y:ys):xs) = y : (combine $ xs ++ [ys])
combine [] = []


instance Arbitrary T where
  arbitrary = liftM5 T arbitrary arbitrary
                       arbitrary arbitrary arbitrary
#ifndef USE_SMARTCHECK
#ifdef USE_CUSTOM_SHRINKER
  shrink t@(T i1 i2 i3 i4 i5) = combine [
        [T j1 j2 j3 j4 j5 | (j1, j2, j3, j4, j5) <- shrink (i1, i2, i3, i4, i5)],
        [T  s i2 i3 i4 i5 | s <- shrinkAdjacentPairs i1],
        [T i1  s i3 i4 i5 | s <- shrinkAdjacentPairs i2],
        [T i1 i2  s i4 i5 | s <- shrinkAdjacentPairs i3],
        [T i1 i2 i3  s i5 | s <- shrinkAdjacentPairs i4],
        [T i1 i2 i3 i4  s | s <- shrinkAdjacentPairs i5],
        [T j1 j2 j3 j4 j5 | [j1, j2, j3, j4, j5] <- reorderListShrinker [i1, i2, i3, i4, i5] ],
        sp1 t, sp2 t, sp3 t, sp4 t
    ]
    where sp1 (T i1 i2 i3 [J n1] [J n2]) = [T i1 i2 i3 [J s1] [J s2] | (s1, s2)  <- customIntShrinker (n1, n2)]
          sp1 _ = []
          sp2 (T i1 i2 [J n1] [J n2] i5) = [T i1 i2 [J s1] [J s2] i5 | (s1, s2)  <- customIntShrinker (n1, n2)]
          sp2 _ = []
          sp3 (T i1 [J n1] [J n2] i4 i5) = [T i1 [J s1] [J s2] i4 i5 | (s1, s2)  <- customIntShrinker (n1, n2)]
          sp3 _ = []
          sp4 (T [J n1] [J n2] i3 i4 i5) = [T [J s1] [J s2] i3 i4 i5 | (s1, s2)  <- customIntShrinker (n1, n2)]
          sp4 _ = []
#else
  shrink (T i1 i2 i3 i4 i5) = [
    T j1 j2 j3 j4 j5 | (j1, j2, j3, j4, j5) <- shrink (i1, i2, i3, i4, i5)]
#endif
#endif

toList :: T -> [[Int16]]
toList (T i0 i1 i2 i3 i4) =
  (map . map) getInt [i0, i1, i2, i3, i4]


pre :: T -> Bool
pre t = all ((< 256) . sum) (toList t)

post :: T -> Bool
post t = (sum . concat) (toList t) < 5 * 256

prop :: T -> Property
prop t = pre t ==> post t

main = evalQuickCheck prop
