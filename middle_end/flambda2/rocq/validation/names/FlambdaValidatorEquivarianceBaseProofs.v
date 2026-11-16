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


(* Lemmas of FlambdaValidatorEquivarianceProofs.v that files below the
   validation import: the two tests of free names in a handler on
   renamed arguments, and the inlining of the calls of symbols known
   to be closures on a renamed term. Moved here as they were; this
   file needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaNormaliseSpec FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaUnusedContsSpec.
Import ListNotations.

Lemma existsb_map_with {A B : Type} (f : B -> bool) (g : A -> B)
    (h : A -> bool) (l : list A) :
  (∀ a : A, f (g a) = h a) -> existsb f (map g l) = existsb h l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

(* Whether a continuation is among those that an [Apply_cont] names. *)
Lemma among_apply_cont_continuations_renamed (r : renaming)
    (k : continuation) (ac : apply_cont) :
  among (rename_continuation r k)
    (apply_cont_continuations (rename_apply_cont r ac))
  = among k (apply_cont_continuations ac).
Proof.
  rewrite apply_cont_continuations_renamed.
  apply among_rename_continuation.
Qed.

Lemma var_in_simple_equivariant (r : renaming) (x : variable) (s : simple) :
  var_in_simple (rename_variable r x) (rename_simple r s) = var_in_simple x s.
Proof. destruct s; simpl; [ apply rename_variable_eqb | | ]; reflexivity. Qed.

Lemma var_in_value_slots_equivariant (r : renaming) (x : variable)
    (slots : list (value_slot * simple)) :
  var_in_value_slots (rename_variable r x)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = var_in_value_slots x slots.
Proof.
  unfold var_in_value_slots. apply existsb_map_with. intros slot.
  apply var_in_simple_equivariant.
Qed.

Lemma var_in_named_equivariant (r : renaming) (x : variable) (d : named) :
  var_in_named (rename_variable r x) (rename_named r d) = var_in_named x d.
Proof.
  destruct d; simpl;
    first
      [ apply var_in_simple_equivariant | apply var_in_simples_equivariant
      | apply var_in_value_slots_equivariant ].
Qed.

Lemma var_in_static_binding_equivariant (r : renaming) (x : variable)
    (binding : static_binding) :
  var_in_static_binding (rename_variable r x)
    (rename_static_binding r binding)
  = var_in_static_binding x binding.
Proof.
  destruct binding as [id c | id | sym const]; simpl; try reflexivity.
  destruct const; simpl;
    first
      [ reflexivity | apply var_in_value_slots_equivariant
      | apply var_in_simples_equivariant | apply var_in_simple_equivariant ].
Qed.

Fixpoint var_free_b_equivariant_aux (r : renaming) (x : variable) (e : expr) :
  var_free_b (rename_variable r x) (rename_expr r e) = var_free_b x e
with var_free_in_handler_b_equivariant (r : renaming) (x : variable)
    (h : continuation_handler) :
  var_free_in_handler_b (rename_variable r x) (rename_handler r h)
  = var_free_in_handler_b x h.
Proof.
  - destruct e
      as [ y d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + rewrite var_in_named_equivariant, rename_variable_eqb,
        var_free_b_equivariant_aux.
      reflexivity.
    + rewrite var_free_b_equivariant_aux. f_equal.
      apply existsb_map_with. intros binding.
      apply var_in_static_binding_equivariant.
    + rewrite var_free_in_handler_b_equivariant, var_free_b_equivariant_aux.
      reflexivity.
    + rewrite var_free_b_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (var_free_in_handler_b_equivariant r x h), IH. reflexivity.
    + rewrite exn_extra_simples_rename, var_in_simple_equivariant,
        !var_in_simples_equivariant.
      reflexivity.
    + apply var_in_simples_equivariant.
    + rewrite var_in_simple_equivariant. f_equal.
      apply existsb_map_with. intros arm. apply var_in_simples_equivariant.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite among_rename_variable, var_free_b_equivariant_aux. reflexivity.
Qed.

Fixpoint cont_free_b_equivariant_aux (r : renaming) (k : continuation)
    (e : expr) :
  cont_free_b (rename_continuation r k) (rename_expr r e) = cont_free_b k e
with cont_free_in_handler_b_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) :
  cont_free_in_handler_b (rename_continuation r k) (rename_handler r h)
  = cont_free_in_handler_b k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl.
    + apply cont_free_b_equivariant_aux.
    + apply cont_free_b_equivariant_aux.
    + rewrite cont_free_in_handler_b_equivariant, rename_continuation_eqb,
        cont_free_b_equivariant_aux.
      reflexivity.
    + rewrite cont_free_b_equivariant_aux. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (cont_free_in_handler_b_equivariant r k h), IH. reflexivity.
    + rewrite !rename_continuation_eqb. reflexivity.
    + exact (among_apply_cont_continuations_renamed r k ac).
    + apply existsb_map_with. intros arm.
      exact (among_apply_cont_continuations_renamed r k (snd arm)).
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply cont_free_b_equivariant_aux.
Qed.

(* ======================================================================== *)
(* Inlining calls of symbols known to be closures                           *)
(* ======================================================================== *)

Lemma symbols_of_renamed (r : renaming) (bindings : list static_binding) :
  ∀ known : list (symbol * option code_id),
    symbols_of (map (rename_static_binding r) bindings) known
    = symbols_of bindings known.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros known; simpl; [ reflexivity | apply IH | apply IH | ].
  destruct const; simpl; apply IH.
Qed.

Lemma call_agrees_b_renamed (r : renaming) (ap : apply_expr) (c : code) :
  call_agrees_b (rename_apply r ap) (rename_code r c) = call_agrees_b ap c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold call_agrees_b. simpl. rewrite length_map. reflexivity.
Qed.

Lemma inline_known_apply_equivariant (r : renaming) (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (ap : apply_expr) :
  inline_known_apply kept (map_values (rename_code r) known_code)
    known_symbols (rename_apply r ap)
  = rename_expr r (inline_known_apply kept known_code known_symbols ap).
Proof.
  unfold inline_known_apply.
  replace (ap_callee (rename_apply r ap)) with (rename_simple r (ap_callee ap))
    by reflexivity.
  replace (ap_call_kind (rename_apply r ap)) with (ap_call_kind ap)
    by reflexivity.
  destruct (ap_callee ap) as [x | s | c]; cbn [rename_simple];
    try reflexivity.
  destruct (ap_call_kind ap) as [called | ]; [ | reflexivity ].
  rewrite lookup_map_values.
  destruct (lookup s known_symbols) as [[code | ] | ]; try reflexivity.
  destruct (lookup called known_code) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite FlambdaEquivarianceProofs.inlinable_b_equivariant,
    call_agrees_b_renamed.
  destruct
    (negb (among called kept)
     && (Pos.eqb code called && inlinable_b ap c && call_agrees_b ap c));
    [ | reflexivity ].
  apply inlined_body_equivariant.
Qed.

Fixpoint inline_known_calls_equivariant_aux (r : renaming)
    (kept : list code_id) (known_symbols : list (symbol * option code_id))
    (e : expr) :
  ∀ known_code : list (code_id * code),
    inline_known_calls kept (map_values (rename_code r) known_code)
      known_symbols (rename_expr r e)
    = rename_expr r (inline_known_calls kept known_code known_symbols e)
with inline_known_calls_in_handler_equivariant (r : renaming)
    (kept : list code_id) (known_symbols : list (symbol * option code_id))
    (h : continuation_handler) :
  ∀ known_code : list (code_id * code),
    inline_known_calls_in_handler kept (map_values (rename_code r) known_code)
      known_symbols (rename_handler r h)
    = rename_handler r
        (inline_known_calls_in_handler kept known_code known_symbols h)
with inline_known_calls_in_binding_equivariant (r : renaming)
    (kept : list code_id) (known_symbols : list (symbol * option code_id))
    (binding : static_binding) :
  ∀ known_code : list (code_id * code),
    inline_known_calls_in_binding kept (map_values (rename_code r) known_code)
      known_symbols (rename_static_binding r binding)
    = rename_static_binding r
        (inline_known_calls_in_binding kept known_code known_symbols binding)
with inline_known_calls_in_code_equivariant (r : renaming)
    (kept : list code_id) (known_symbols : list (symbol * option code_id))
    (c : code) :
  ∀ known_code : list (code_id * code),
    inline_known_calls_in_code kept (map_values (rename_code r) known_code)
      known_symbols (rename_code r c)
    = rename_code r
        (inline_known_calls_in_code kept known_code known_symbols c).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known_code;
      [ | | | | exact
                  (inline_known_apply_equivariant r kept known_code
                     known_symbols ap)
      | | | ];
      simpl; try reflexivity.
    + rewrite inline_known_calls_equivariant_aux. reflexivity.
    + (* Let_static: what is known in the body is what the group, as
         rewritten, makes known *)
      assert (Hbindings :
        map
          (inline_known_calls_in_binding kept
             (map_values (rename_code r) known_code) known_symbols)
          (map (rename_static_binding r) bindings)
        = map (rename_static_binding r)
            (map (inline_known_calls_in_binding kept known_code known_symbols)
               bindings)).
      { induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite (inline_known_calls_in_binding_equivariant r kept
                   known_symbols binding), IH.
        reflexivity. }
      rewrite Hbindings, codes_of_equivariant, symbols_of_renamed,
        inline_known_calls_equivariant_aux.
      reflexivity.
    + rewrite inline_known_calls_in_handler_equivariant,
        inline_known_calls_equivariant_aux.
      reflexivity.
    + rewrite inline_known_calls_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (inline_known_calls_in_handler_equivariant r kept known_symbols
                 h),
        IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known_code. simpl.
    rewrite inline_known_calls_equivariant_aux. reflexivity.
  - destruct binding as [id c | id | sym const]; intros known_code; simpl;
      try reflexivity.
    rewrite inline_known_calls_in_code_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body].
    intros known_code. simpl.
    rewrite inline_known_calls_equivariant_aux. reflexivity.
Qed.
