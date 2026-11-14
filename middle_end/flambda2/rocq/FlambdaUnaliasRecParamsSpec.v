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

(* A parameter of a loop that every turn passes on as it is, and that
   the loop is given one variable for, is that variable: a
   normalisation that is an equivalence.

   This file is to be reviewed: [unalias_rec_param], [unalias_rec_from]
   and [unalias_rec_params], which are extracted and run by the
   validation of middle_end/flambda2/tests/semantics/
   (FlambdaValidateSpec.tidy_expr_before and tidy_expr_after apply the
   pass), and the statement.

   FlambdaUnaliasParamsSpec.unalias_params takes away a parameter of a
   non-recursive continuation that every application gives one
   variable. Simplify keeps, for a function whose self tail calls it
   has made jumps, a parameter that the function passes on unchanged:
   of

     let rec loop acc n = if n <= 0 then acc else loop acc (n - 1)

   the recursive continuation that the function has become has both
   parameters, each turn gives the first one itself, and the entry
   gives it the variable from outside. This is the same removal for
   that continuation.

   [unalias_rec_param], at a recursive [Let_cont] that binds one
   continuation [k], for one position: where every application of [k]
   in the body of the [Let_cont] gives one variable [x] that no binder
   between the [Let_cont] and the application binds
   (FlambdaUnaliasParamsSpec.all_give_var), and every application of
   [k] in the handler gives the parameter [p] itself, which no binder
   between the head of the handler and the application binds, the
   parameter is taken away: every application loses the argument, and
   the handler begins with a [Let] of [p] to [x], with the kind that
   the handler had for the parameter. The other tests: the handler
   is no exception handler, and neither [x] nor [p] is one of the
   other parameters. [x] may be [p]: the loop that a function has
   become has the names of the parameters of the function for its
   own, and is given, from outside, the variable of the same name;
   the [Let] at the head of the handler then binds [p] to the
   variable [p] from outside, which the parameter hid. As for
   FlambdaUnaliasParamsSpec.unalias_param, which does not take the
   case of one name, every mention of [k] is then an application of
   it: no call returns or raises to [k], no trap action names it and
   no [Let_cont] inside binds [k] again. A recursive [Let_cont] that
   binds more than one continuation is left as it is: another of its
   handlers may apply [k] where [p] is not in scope.

   Why it is an equivalence. The handler of [k] runs under the
   variables of the [Let_cont], with its parameters bound. At every
   entry the parameter holds what [x] held where the [Let_cont] was
   run: at an entry from outside because the application gives [x],
   which no binder on the way has bound; at an entry from the handler
   because the application gives [p], which no binder on the way has
   bound, and [p] held that value when the handler was entered. So the
   [Let] at the head of the handler, which reads [x] under the
   variables of the [Let_cont], [x] being none of the parameters that
   are left, binds [p] to the value that the application gave. The
   machine tests the kind of
   a value where a handler is applied, and the [Let] tests the same
   kind of the same value: the two runs stop or go on together at
   every entry.

   The pass is that step for each position, the last first
   ([unalias_rec_from]), at every recursive [Let_cont] of the term,
   its body first; it goes into the handler of a non-recursive
   continuation. The handlers of recursive continuations and the
   bodies of code are left as they are, as by
   FlambdaUnaliasParamsSpec.unalias_params.

   Not done: an application in the handler that gives [x], the
   variable from outside, which would be as sound; a parameter that
   the loop is given one symbol or one constant for
   (FlambdaBindConstParamsSpec.v does it for a non-recursive
   continuation); a group of more than one continuation.

   The proofs are in FlambdaUnaliasRecParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaUnaliasParamsSpec.
Import ListNotations.

Definition unalias_rec_param (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (continuation_handler * expr) :=
  match h with
  | Handler params metadata handler =>
    match nth_error params i, nth_error (hm_params_kinds metadata) i with
    | Some p, Some kind =>
      match first_given_var k i body with
      | Some (Some x) =>
        if negb (hm_is_exn_handler metadata)
           && all_give_var k i x true body
           && all_give_var k i p true handler
           && negb (among x (drop_nth i params))
           && negb (among p (drop_nth i params))
        then
          Some
            (Handler (drop_nth i params)
               {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
                  hm_is_exn_handler := false |}
               (Let p kind (Simple (Var x)) (drop_arg k i handler)),
             drop_arg k i body)
        else None
      | _ => None
      end
    | _, _ => None
    end
  end.

(* The same for every position below [i], the last first. *)
Fixpoint unalias_rec_from (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : continuation_handler * expr :=
  match i with
  | O => (h, body)
  | S i =>
    match unalias_rec_param i k h body with
    | Some (h', body') => unalias_rec_from i k h' body'
    | None => unalias_rec_from i k h body
    end
  end.

(* The pass: every recursive [Let_cont] of a term that binds one
   continuation, its body first. *)
Fixpoint unalias_rec_params (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (unalias_rec_params body)
  | Let_static bindings body =>
    Let_static bindings (unalias_rec_params body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (unalias_rec_params handler))
      (unalias_rec_params body)
  | Let_cont_recursive handlers body =>
    let body := unalias_rec_params body in
    match handlers with
    | [(k, Handler params metadata handler)] =>
      let '(h, body) :=
        unalias_rec_from (length params) k
          (Handler params metadata handler) body
      in
      Let_cont_recursive [(k, h)] body
    | _ => Let_cont_recursive handlers body
    end
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result behaves as the term does, under all environments. *)
Definition unalias_rec_params_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (unalias_rec_params e).
