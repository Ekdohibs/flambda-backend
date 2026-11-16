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

(* Proof of the statement of FlambdaNeverReturnsScopeSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaWellFormedSpec FlambdaNormaliseSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaNeverReturnsSpec FlambdaNeverReturnsScopeSpec.
Import ListNotations.

(* A call under a binding of its return continuation to a handler that
   is [Invalid]: nothing is free in it that is not free in the call. *)
Lemma scoped_cut_return (known : list (code_id * code)) (ap : apply_expr) :
  scoped (Apply ap) (cut_return known ap).
Proof.
  unfold cut_return.
  destruct (ap_call_kind ap) as [called | ]; [ | apply scoped_refl ].
  destruct (lookup called known) as [c | ]; [ | apply scoped_refl ].
  destruct (invalid_params c (length (unarize (ap_return_arity ap))))
    as [params | ];
    [ | apply scoped_refl ].
  destruct (never_returns_b ap c); [ | apply scoped_refl ].
  intros _. scope.
  split; [ reflexivity | ]. split.
  - apply incl_refl.
  - intros k Hk. cbn [app] in Hk. apply in_without in Hk. exact (proj1 Hk).
Qed.

Lemma map_fst_cut_returns (known : list (code_id * code))
    (handlers : list (continuation * continuation_handler)) :
  map fst (map (fun '(k, h) => (k, cut_returns_in_handler known h)) handlers)
  = map fst handlers.
Proof. rewrite map_map. apply map_ext. intros [k h]. reflexivity. Qed.

Fixpoint scoped_cut_returns (e : expr) :
  ∀ known : list (code_id * code), scoped e (cut_returns known e)
with scoped_cut_returns_handler (h : continuation_handler) :
  ∀ known : list (code_id * code),
    scoped_handler h (cut_returns_in_handler known h)
with scoped_cut_returns_code (c : code) :
  ∀ known : list (code_id * code),
    scoped_code c (cut_returns_in_code known c).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; cbn [cut_returns]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply scoped_cut_returns ].
    + apply scoped_let_static; [ | apply scoped_cut_returns ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | s const];
        cbn [cut_returns_in_binding];
        [ apply scoped_bindings_cons_code;
          [ apply scoped_cut_returns_code | exact IH ]
        | apply scoped_bindings_cons_same; exact IH
        | apply scoped_bindings_cons_same; exact IH ].
    + destruct (returns_to_invalid k h body); [ apply scoped_refl | ].
      apply scoped_let_cont;
        [ apply scoped_cut_returns_handler | apply scoped_cut_returns ].
    + apply scoped_let_cont_recursive;
        [ apply map_fst_cut_returns | | apply scoped_cut_returns ].
      induction handlers as [ | [k h] handlers IH]; cbn [map];
        [ apply scoped_handlers_nil | ].
      apply scoped_handlers_cons;
        [ apply scoped_cut_returns_handler | exact IH ].
    + apply scoped_cut_return.
  - destruct h as [params metadata handler]. intros known.
    cbn [cut_returns_in_handler].
    apply scoped_handler_intro, scoped_cut_returns.
  - destruct c as [params my_closure ret exn metadata body]. intros known.
    cbn [cut_returns_in_code].
    apply scoped_code_intro, scoped_cut_returns.
Qed.

Theorem cut_returns_keeps_scope : cut_returns_keeps_scope_statement.
Proof.
  intros known. apply keeps_scope_intro. intros e.
  apply scoped_cut_returns.
Qed.
