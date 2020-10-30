{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Support
import EvalCommon
import Test.QuickCheck
import Support
import Data.Char

#ifdef USE_SMARTCHECK
import Test.SmartCheck
#endif


#ifdef USE_SMARTCHECK
instance SubTypes Lang
instance SubTypes Var
  where baseType _ = True
instance SubTypes Mod
instance SubTypes Func
instance SubTypes Stmt
instance SubTypes Exp
#endif

instance Arbitrary Var where
  arbitrary = Var <$> suchThat arbitrary
    (\s -> all isAlphaNum s && not (null s))


instance Arbitrary Lang where
  arbitrary = Lang <$> nonEmpty arbitrary <*> nonEmpty arbitrary
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

instance Arbitrary Mod where
  arbitrary = Mod <$> nonEmpty arbitrary <*> nonEmpty arbitrary
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

instance Arbitrary Func where
  arbitrary = Func <$> arbitrary <*> nonEmpty arbitrary <*> nonEmpty arbitrary
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

instance Arbitrary Stmt where
  arbitrary = do
    v  <- arbitrary
    e  <- arbitrary
    let a0 = Assign v e
    let a1 = Alloc v e
    let a2 = Return e
    elements [a0, a1, a2]
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

instance Arbitrary Exp where
  arbitrary = go
    where
    go = go' =<< choose (0::Int, 100)
    go' 0 = oneof [Bool <$> arbitrary, Int <$> arbitrary]
    go' i = let g = go' =<< choose (0::Int, i-1) in
            frequency [ (10, Not <$> g)
                      , (100, And <$> g <*> g)
                      , (100, Or  <$> g <*> g)
                      , (100, Add  <$> g <*> g)
                      , (100, Sub  <$> g <*> g)
                      , (100, Mul  <$> g <*> g)
                      , (100, Div  <$> g <*> g)
                      ]
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

main = evalQuickCheck prop_parse
