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



(* What the answers of reification mean.

   This file is to be reviewed: what an answer is said to say of a
   value ([reification_holds]) and the statement. Reification is
   FlambdaReify.v, which is to be read against
   middle_end/flambda2/types/reify.ml; the provers that it asks are
   FlambdaProvers.v, with their statements in FlambdaProversSpec.v,
   whose header says what a typing environment, a valuation and a
   well-formed type are here.

   What the compiler does with an answer is
   [Reification.try_to_reify] (simplify/lifting/reification.ml), which
   simplify/simplify_named.ml calls on the result of a primitive that
   asks for it (the flag [try_reify] of [Simplify_primitive_result]),
   with the type that the environment has for the variable being
   bound.
   - [Simple s]: the term becomes [s], and the variable an alias of
     it; but if [s] is the variable being bound the term is left as it
     is, so that no binding is of itself.
   - [Lift]: if the variable is bound at the normal name mode and
     lifting is allowed there, a static constant is made of the answer
     ([create_static_const]) and bound to a symbol, a new one or one
     that an equal constant has already ([lift]), and the term becomes
     that symbol. Otherwise the term is left as it is.
   - [Invalid]: the term is invalid.
   - [Cannot_reify]: the term is left as it is.

   What is said, of a type, a typing environment and a valuation that
   satisfies the environment strictly.
   - [Simple s]: every value of the type is the value of [s] in the
     valuation. What FlambdaProversSpec.v says of a [Simple] that a
     prover answers holds of this one: that a term may use it is not
     said here.
   - [Lift (Immutable_block tag _ shape fields)]: every value of the
     type is a block of that tag and of that shape, whose fields are
     the values of the [Simple]s answered, all of them and in their
     order.
   - [Lift (Boxed_number k n)]: every value of the type is a boxed
     number of that kind, whose contents are the number [n].
   - [Invalid]: the type has no value.
   Nothing is said of [Cannot_reify], nor of [None], which is a stop
   of the compiler.

   What is not said.
   - That the value may be replaced by a symbol bound to a block or a
     boxed number made of what is answered. A block and a symbol bound
     to an equal block are two values of the model, so that the
     replacement keeps the meaning of the program is for the
     statement of the rewrite that lifts (decision D6).
   - Anything of [is_unique], or of the conditions under which the
     compiler lets a variable be a field of what is lifted
     ([Reify.var_allowed]: the environment in which the variable must
     be defined, the variables defined at toplevel, the symbol
     projections, the allocation mode and
     [Provers.never_holds_locally_allocated_values]). They are
     transcribed; allocation modes and scopes of lifting have no
     meaning in the denotation of types (FlambdaTypeMeaningSpec.v).
   - Anything under the notion of satisfaction that exempts the names
     that hold poison: reification trusts what the environment says
     of a name, as the provers do (discrepancy T6-1).

   The type is to be well formed ([well_formed_input] of
   FlambdaProversSpec.v), for the numbers of the immediates of a
   variant and of a type of naked immediates, which
   [Provers.meet_naked_immediates] reads.

   The proofs are in FlambdaReifyProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaExpandHead FlambdaProvers
  FlambdaProversSpec FlambdaReify.
Import ListNotations.

(* What an answer of reification says of a value. *)
(* CR mshinwell: nothing here says that lifting is right. The answer
   [Lift] says what the value is, not that a static constant made of
   it may take its place: the model tells a block from a symbol bound
   to an equal block, and it has no allocation modes, which are what
   [Reify.var_allowed] guards. It would be wrong to read the
   statement as the soundness of what [Reification.try_to_reify] does
   with [Lift]. Decision D6. *)
Definition reification_holds (statics : static_env) (valuation : env)
    (result : Reification_result.t) (v : value) : Prop :=
  match result with
  | Reification_result.Simple s => eval_simple valuation s = Some v
  | Reification_result.Lift (To_lift.Immutable_block tag _ shape fields) =>
    ∃ field_values : list value,
      block_of statics v = Some (tag, field_values)
      ∧ block_has_shape (Scannable shape) tag field_values
      ∧ eval_simples valuation fields = Some field_values
  | Reification_result.Lift (To_lift.Boxed_number k n) =>
    boxed_number_of statics v
    = Some (const_of_naked_number (Reify.unboxed_number_kind k) n)
  | Reification_result.Cannot_reify => True
  | Reification_result.Invalid => False
  end.

(* [Reify.reify], whatever the conditions for lifting a variable. *)
Definition reify_sound_statement : Prop :=
  ∀ (allowed_if_free_vars_defined_in : Typing_env.t)
    (var_is_defined_at_toplevel var_is_symbol_projection : variable -> bool)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (result : Reification_result.t) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    well_formed_input tenv t ->
    type_satisfied statics valuation v t ->
    Reify.reify allowed_if_free_vars_defined_in var_is_defined_at_toplevel
      var_is_symbol_projection tenv t
    = Some result ->
    reification_holds statics valuation result v.

(* The kind of naked numbers of a boxed number is its unboxed kind. *)
Definition unboxed_number_kind_statement : Prop :=
  ∀ k : boxable_number,
    Naked_number (Reify.unboxed_number_kind k) = unboxed_kind k.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* Reification in the empty environment, of types made with the
   creation functions of FlambdaTypeGrammar.v. No variable is in the
   types, so the conditions for lifting one are not asked. *)
Definition reify_in_empty_env (t : flambda_type)
    : option Reification_result.t :=
  Reify.reify Typing_env.create (fun _ => false) (fun _ => false)
    Typing_env.create t.

(* A tagged immediate that is an alias of its constant, and one that
   is a variant of one immediate, are that constant. A block of two
   fields that are constants is lifted with those fields, and a boxed
   float of one number with that number. A block of which a field is
   not known as a [Simple], and a block of which nothing is known, are
   not reified. The type with no value is invalid. *)
Definition example_reify_statement : Prop :=
  reify_in_empty_env (this_tagged_immediate 3%Z)
  = Some (Reification_result.Simple (Const (Tagged_immediate 3%Z)))
  ∧ reify_in_empty_env
      (tag_immediate
         (Type_naked_immediate
            (Ok
               (No_alias
                  (Head_of_kind_naked_immediate (Or_unknown.Known [3%Z])
                     [])))))
    = Some (Reification_result.Simple (Const (Tagged_immediate 3%Z)))
  ∧ reify_in_empty_env
      (immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
         [this_tagged_immediate 1%Z; this_tagged_immediate 2%Z])
    = Some
        (Reification_result.Lift
           (To_lift.Immutable_block 0%Z false Value_only
              [Const (Tagged_immediate 1%Z); Const (Tagged_immediate 2%Z)]))
  ∧ reify_in_empty_env
      (box_number Boxable_naked_float
         (Type_naked_number Kind_naked_float (Ok (No_alias [5%Z])))
         Alloc_mode.Heap)
    = Some
        (Reification_result.Lift
           (To_lift.Boxed_number Boxable_naked_float 5%Z))
  ∧ reify_in_empty_env
      (immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
         [this_tagged_immediate 1%Z; any_block])
    = Some Reification_result.Cannot_reify
  ∧ reify_in_empty_env any_block = Some Reification_result.Cannot_reify
  ∧ reify_in_empty_env (bottom Value) = Some Reification_result.Invalid.
