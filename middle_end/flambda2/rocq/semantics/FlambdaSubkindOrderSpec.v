(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* The order on subkinds: one says at least as much as another.

   This file is to be reviewed: the definitions and the statements.

   A subkind at a parameter is a promise about the value bound.
   Decision D25, at parameters, is that a run is undefined where a
   parameter binds a value that is not of what it states; the test is
   that of FlambdaSubkindSpec.v ([value_at], [values_of_subkinds]).
   Where [a] says at least as much as [b], every value that the test
   accepts at [a] it accepts at [b]: that is the third statement, and
   it is what the order is for. It is for the comparison of the
   validation: where two paired binders are compared, the term before
   is to say at least as much as the term after; the term after may
   promise less, and is then defined wherever the term before is.

   The order is read off the two subkinds and is not the largest with
   that property. What it has:
   - any subkind says at least as much as [Subkind_anything];
   - a tagged immediate says at least as much as a tagged immediate,
     and so does a variant with no constructor that is a block: the
     test of a variant accepts a constant constructor only where its
     number is in the range of a tagged immediate;
   - a boxed number, as much as a boxed number of the same kind;
   - a variant, as much as a variant that has its constant
     constructors and, for each of its tags, the tag: with the fields
     not given, or, where both give them, with as many fields, each
     saying no more. Of the entries of the first variant only the
     first for a tag is read, as the test reads it;
   - a field or a parameter that may not be null says at least as much
     as the same one that may.
   For lists of parameters the convention of the metadata is kept: a
   list that is shorter says nothing of the parameters that are left.
   So any list says at least as much as a shorter one of which it says
   at least as much entry by entry, and a shorter list says at least
   as much as a longer one only if the entries that are left say
   nothing ([says_nothing]: anything, and it may be null).

   The statements: the order on lists is reflexive and transitive, and
   what the test of the machine accepts at a list it accepts at any
   list of which that list says at least as much.

   The proofs are in FlambdaSubkindOrderProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSubkindSpec.
Import ListNotations.

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

(* The first entry of [shapes] for a tag: [Some None] where its fields
   are not given. *)
Fixpoint shape_of_tag (tag : Z) (shapes : constructor_shapes)
    : option (option field_subkinds) :=
  match shapes with
  | Shapes_nil => None
  | Shape_undetermined tag' rest =>
    if Z.eqb tag tag' then Some None else shape_of_tag tag rest
  | Shape_determined tag' sks rest =>
    if Z.eqb tag tag' then Some (Some sks) else shape_of_tag tag rest
  end.

(* [a] says at least as much as [b]. For the shapes of a variant,
   [seen] has the tags of the entries gone past: an entry for one of
   them is not the first for its tag, and is not read. *)
Fixpoint subkind_says_at_least (a b : subkind) {struct a} : bool :=
  match b with
  | Subkind_anything => true
  | Subkind_tagged_immediate =>
    match a with
    | Subkind_tagged_immediate => true
    | Subkind_variant _ Shapes_nil => true
    | Subkind_anything | Subkind_boxed_number _ | Subkind_variant _ _ =>
      false
    end
  | Subkind_boxed_number n' =>
    match a with
    | Subkind_boxed_number n => boxable_number_eqb n n'
    | Subkind_anything | Subkind_tagged_immediate | Subkind_variant _ _ =>
      false
    end
  | Subkind_variant consts' shapes' =>
    match a with
    | Subkind_variant consts shapes =>
      forallb (fun i => existsb (Z.eqb i) consts') consts
      && shapes_say_at_least [] shapes shapes'
    | Subkind_anything | Subkind_tagged_immediate | Subkind_boxed_number _ =>
      false
    end
  end
with shapes_say_at_least (seen : list Z) (shapes shapes' : constructor_shapes)
    {struct shapes} : bool :=
  match shapes with
  | Shapes_nil => true
  | Shape_undetermined tag rest =>
    (existsb (Z.eqb tag) seen
     || match shape_of_tag tag shapes' with
        | Some None => true
        | Some (Some _) | None => false
        end)
    && shapes_say_at_least (tag :: seen) rest shapes'
  | Shape_determined tag sks rest =>
    (existsb (Z.eqb tag) seen
     || match shape_of_tag tag shapes' with
        | Some None => true
        | Some (Some sks') => fields_say_at_least sks sks'
        | None => false
        end)
    && shapes_say_at_least (tag :: seen) rest shapes'
  end
with fields_say_at_least (sks sks' : field_subkinds) {struct sks} : bool :=
  match sks, sks' with
  | Fields_nil, Fields_nil => true
  | Fields_cons nullable sk rest, Fields_cons nullable' sk' rest' =>
    (negb nullable || nullable')
    && subkind_says_at_least sk sk'
    && fields_say_at_least rest rest'
  | _, _ => false
  end.

(* The same for what a parameter states: whether it may be null, and
   its subkind. *)
Definition param_says_at_least (a b : param_subkind) : bool :=
  (negb (fst a) || fst b) && subkind_says_at_least (snd a) (snd b).

(* The same for the lists of two binders. A list that is shorter says
   nothing of the parameters that are left. *)
Fixpoint params_say_at_least (a b : list param_subkind) {struct b} : bool :=
  match b with
  | [] => true
  | pb :: b' =>
    match a with
    | [] => says_nothing (fst pb) (snd pb) && params_say_at_least [] b'
    | pa :: a' => param_says_at_least pa pb && params_say_at_least a' b'
    end
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition params_say_at_least_refl_statement : Prop :=
  ∀ a : list param_subkind, params_say_at_least a a = true.

Definition params_say_at_least_trans_statement : Prop :=
  ∀ a b c : list param_subkind,
    params_say_at_least a b = true ->
    params_say_at_least b c = true ->
    params_say_at_least a c = true.

(* What the test of the machine accepts at a list, it accepts at any
   list of which that list says at least as much. *)
Definition values_of_subkinds_weaken_statement : Prop :=
  ∀ (statics : static_env) (vs : list value) (a b : list param_subkind),
    params_say_at_least a b = true ->
    values_of_subkinds statics vs a = true ->
    values_of_subkinds statics vs b = true.
