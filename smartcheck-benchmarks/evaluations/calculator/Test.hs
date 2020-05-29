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

import EvalCommon

#ifdef USE_SMARTCHECK
import Test.SmartCheck
#endif

-----------------------------------------------------------------

data Exp = C Int
         | Add Exp Exp
         | Div Exp Exp
  deriving (Show, Read, Typeable, Generic)

#ifdef USE_SMARTCHECK
instance SubTypes Exp
#endif

eval :: Exp -> Maybe Int
eval (C i) = Just i
eval (Add e0 e1) =
  liftM2 (+) (eval e0) (eval e1)
eval (Div e0 e1) =
  let e = eval e1 in
  if e == Just 0 then Nothing
    else liftM2 div (eval e0) e

instance Arbitrary Exp where
  arbitrary = sized mkM
    where
    mkM 0 = liftM C arbitrary
    mkM n = oneof [ liftM2 Add mkM' mkM'
                  , liftM2 Div mkM' mkM' ]
      where mkM' = mkM =<< choose (0,n-1)
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

-- property: so long as 0 isn't in the divisor, we won't try to divide by 0.
-- It's false: something might evaluate to 0 still.
prop_div :: Exp -> Property
prop_div e = divSubTerms e ==> eval e /= Nothing

  -- precondition: no dividand in a subterm can be 0.
divSubTerms :: Exp -> Bool
divSubTerms (C _)         = True
divSubTerms (Div _ (C 0)) = False
divSubTerms (Add e0 e1)   = divSubTerms e0 && divSubTerms e1
divSubTerms (Div e0 e1)   = divSubTerms e0 && divSubTerms e1

-- Get the minimal offending sub-value.
findVal :: Exp -> (Exp,Exp)
findVal (Div e0 e1)
  | eval e1 == Just 0     = (e0,e1)
  | eval e1 == Nothing    = findVal e1
  | otherwise             = findVal e0
findVal a@(Add e0 e1)
  | eval e0 == Nothing    = findVal e0
  | eval e1 == Nothing    = findVal e1
  | eval a == Just 0      = (a,a)
findVal _                 = error "not possible"

size :: Exp -> Int
size e = case e of
  C _       -> 1
  Add e0 e1 -> 1 + size e0 + size e1
  Div e0 e1 -> 1 + size e0 + size e1

main = evalQuickCheck prop_div
