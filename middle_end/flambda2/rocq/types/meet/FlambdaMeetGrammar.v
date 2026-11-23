(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                       Pierre Chambart, OCamlPro                        *)
(*           Mark Shinwell and Leo White, Jane Street Europe              *)
(*                                                                        *)
(*   Copyright 2013--2017 OCamlPro SAS                                    *)
(*   Copyright 2014--2017 Jane Street Group LLC                           *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* The header above is that of
   middle_end/flambda2/term_basics/simple.ml,
   the file of the compiler that this file transcribes. *)



(* What meet uses of its neighbours, and that FlambdaTypingEnv.v and
   FlambdaMapSpec.v do not have.

   This file is to be reviewed, each definition against the function of
   the compiler that it names. It holds definitions only and it
   extracts. They are:
   - [kind_of_simple], after [Simple.kind] of
     middle_end/flambda2/term_basics/simple.ml, the one function of
     the compiler that this file transcribes. It is here for
     FlambdaTypingEnv.v, which is not this stream's to edit, and where
     it is to move under the same name;
   - [sort_numbers], a sort of integers, which the module of maps
     has not and is not to have; it is after no function of the
     compiler.
   What was here of the grammar of types ([must_be_singleton] and the
   others) is in FlambdaTypeGrammar.v.

   The difference from the compiler: [kind_of_simple] is
   [Simple.kind]. A variable of the compiler carries its kind; a
   variable of the model does not, and its kind is that of the type
   that the environment has for it (FlambdaTypingEnv.v). [None] where
   the environment has none. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv.
Import ListNotations.

(* ======================================================================== *)
(* Of the module of maps                                                    *)
(* ======================================================================== *)

(* Integers in increasing order. *)
Fixpoint insert_number (i : Z) (sorted : list Z) : list Z :=
  match sorted with
  | [] => [i]
  | j :: rest => if Z.leb i j then i :: sorted else j :: insert_number i rest
  end.

Definition sort_numbers (numbers : list Z) : list Z :=
  fold_right insert_number [] numbers.

(* ======================================================================== *)
(* Of the typing environment                                                *)
(* ======================================================================== *)

(* [Simple.kind], read from the environment for a variable. *)
Definition kind_of_simple (tenv : Typing_env.t) (simple : simple)
    : option flambda_kind :=
  match simple with
  | Const c => Some (kind_of_const c)
  | Symbol _ => Some Value
  | Var x => option_map kind (Typing_env.find tenv (Name_var x) None)
  end.
