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

(* Comparing two expressions up to their bound names, where the first
   may state more of its parameters than the second.

   This file is to be reviewed: the definitions, against
   FlambdaCompareSpec.expr_eqb, of which [expr_says_at_least] is a
   copy but for the metadata of two paired binders; and the
   statements.

   The machine has no step where a handler is applied, or the body
   of code entered, with a value that is not of what the parameter
   states (decision D25, at parameters;
   FlambdaSemanticsSpec.apply_cont_value, call_code).
   FlambdaCompareSpec.expr_eqb asks that two paired binders state the
   same. Simplify states less in places: it gives no subkind to a
   parameter that replaces another. So the validation needs a
   comparison that lets the term after state less.

   Which way round, and why. The first expression is the one that
   may state more, and the second the one that may state less: for
   the validation, the term before Simplify and the term after. In
   [expr_refines e e'] the first is the source and the second the
   target, and it is the target that refines the source: so the
   statement is that the second, which may state less, refines the
   first. A binder that states more may refuse more values, so the
   first has no more defined runs than the second. And every
   constraint of the second is implied by those of the first on the
   same values, an entry that is missing being no constraint: a
   value that passes the first's entry cannot fail the second's, so
   where the first runs the second goes on as it does. Nothing is
   asked of the states where the first is undefined. The other way
   round it fails: the expression that states more would have to run
   wherever the other does, and a test of its own may stop it there.

   [expr_says_at_least vm km e e'] is [expr_eqb vm km e e'] with one
   change. Where [expr_eqb] asks that the metadata of two handlers, or
   of two pieces of code, be the same, it asks that the kinds, the
   arities and the flags be the same and that the first state at least
   as much as the second (FlambdaSubkindOrderSpec.params_say_at_least):
   of the parameters of a handler, of the parameters of code, and of
   its results. Everything else is compared as [expr_eqb] compares it,
   by the same functions.

   What is owed of it:
   - it is sound for refinement, the second expression refining the
     first ([expr_says_at_least_sound_statement], which is
     [expr_refines e e'] with the first as the source): what the
     second states of a value is implied by what the first states of
     it (FlambdaSubkindOrderSpec.values_of_subkinds_weaken_statement),
     so a test that the first passes the second cannot fail. The two
     conventions of the lists hold on both sides: an entry beyond the
     values given is not read, and a value with no entry is not
     tested. It is not sound the other way round, and it is not an
     equivalence: the second may run where the first is undefined;
   - it accepts what [expr_eqb] accepts
     ([expr_eqb_says_at_least_statement]), the order being reflexive.
   What the results of code state is compared, the same way round,
   and the machine does not test the values that code returns against
   it: it reads the arity of the results alone. So that part is a
   filter of the comparison, which may refuse a change that is
   harmless, and the soundness does not rest on it.

   The proofs are in FlambdaCompareOrderProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSubkindOrderSpec FlambdaCompareSpec.
Import ListNotations.

(* ======================================================================== *)
(* The metadata of two paired binders                                       *)
(* ======================================================================== *)

(* The same kinds and the same flag, and [m] states of the parameters
   at least as much as [m']. *)
Definition handler_metadata_says_at_least (m m' : handler_metadata) : bool :=
  kinds_eqb (hm_params_kinds m) (hm_params_kinds m')
  && params_say_at_least (hm_params_subkinds m) (hm_params_subkinds m')
  && Bool.eqb (hm_is_exn_handler m) (hm_is_exn_handler m').

(* The same arities and the same flags, and [m] states of the
   parameters, and of the results, at least as much as [m']. *)
Definition code_metadata_says_at_least (m m' : code_metadata) : bool :=
  arity_eqb (cm_params_arity m) (cm_params_arity m')
  && params_say_at_least (cm_params_subkinds m) (cm_params_subkinds m')
  && arity_eqb (cm_result_arity m) (cm_result_arity m')
  && params_say_at_least (cm_result_subkinds m) (cm_result_subkinds m')
  && Bool.eqb (cm_is_tupled m) (cm_is_tupled m')
  && Bool.eqb (cm_is_stub m) (cm_is_stub m').

(* ======================================================================== *)
(* The comparison                                                           *)
(* ======================================================================== *)

(* FlambdaCompareSpec.expr_eqb, with the two functions above where it
   has [handler_metadata_eqb] and [code_metadata_eqb]. *)
Fixpoint expr_says_at_least (vm km : name_map) (e e' : expr) {struct e}
    : bool :=
  match e, e' with
  | Let x kind d body, Let x' kind' d' body' =>
    kind_eqb kind kind' && named_eqb vm d d'
    && expr_says_at_least ((x, x') :: vm) km body body'
  | Let_static bindings body, Let_static bindings' body' =>
    forall2b (binding_says_at_least vm) bindings bindings'
    && expr_says_at_least vm km body body'
  | Let_cont_non_recursive k h body, Let_cont_non_recursive k' h' body' =>
    handler_says_at_least vm km h h'
    && expr_says_at_least vm ((k, k') :: km) body body'
  | Let_cont_recursive handlers body, Let_cont_recursive handlers' body' =>
    match combine_names (map fst handlers) (map fst handlers') km with
    | Some km' =>
      forall2b (fun '(_, h) '(_, h') => handler_says_at_least vm km' h h')
        handlers handlers'
      && expr_says_at_least vm km' body body'
    | None => false
    end
  | Apply ap, Apply ap' => apply_eqb vm km ap ap'
  | Apply_cont ac, Apply_cont ac' => apply_cont_eqb vm km ac ac'
  | Switch scrutinee arms, Switch scrutinee' arms' =>
    simple_eqb vm scrutinee scrutinee'
    && forall2b
         (fun arm arm' =>
            Z.eqb (fst arm) (fst arm')
            && apply_cont_eqb vm km (snd arm) (snd arm'))
         arms arms'
  | Invalid, Invalid => true
  | _, _ => false
  end
with handler_says_at_least (vm km : name_map) (h h' : continuation_handler)
    {struct h} : bool :=
  match h, h' with
  | Handler params metadata handler, Handler params' metadata' handler' =>
    match bind_names params params' vm with
    | Some vm' =>
      handler_metadata_says_at_least metadata metadata'
      && expr_says_at_least vm' km handler handler'
    | None => false
    end
  end
with binding_says_at_least (vm : name_map)
    (binding binding' : static_binding) {struct binding} : bool :=
  match binding, binding' with
  | Bind_code id c, Bind_code id' c' =>
    Pos.eqb id id' && code_says_at_least c c'
  | Bind_deleted_code id, Bind_deleted_code id' => Pos.eqb id id'
  | Bind_symbol s const, Bind_symbol s' const' =>
    Pos.eqb s s' && static_const_eqb vm const const'
  | _, _ => false
  end
with code_says_at_least (c c' : code) {struct c} : bool :=
  match c, c' with
  | Code params my_closure ret exn metadata body,
    Code params' my_closure' ret' exn' metadata' body' =>
    match bind_names params params' [(my_closure, my_closure')] with
    | Some vm =>
      code_metadata_says_at_least metadata metadata'
      && expr_says_at_least vm [(ret, ret'); (exn, exn')] body body'
    | None => false
    end
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* Under all environments the second refines the first: [expr_refines]
   has the source first. *)
Definition expr_says_at_least_sound_statement : Prop :=
  ∀ e e' : expr, expr_says_at_least [] [] e e' = true -> expr_refines e e'.

(* What [expr_eqb] accepts, it accepts. *)
Definition expr_eqb_says_at_least_statement : Prop :=
  ∀ (vm km : name_map) (e e' : expr),
    expr_eqb vm km e e' = true -> expr_says_at_least vm km e e' = true.
