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

(* Proofs of the statement of FlambdaBindConstParamsSpec.v. Nothing
   here needs review.

   One parameter of one continuation is the site of
   FlambdaUnaliasParamsProofs.v, which is proved for a parameter that
   every application gives one simple: the check of this pass is its
   check for a symbol or a constant, which no binder binds
   ([all_give_plain]). The pass is then by the context lemmas of
   FlambdaContextProofs.v, as there. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNormaliseProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaBindConstParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The check is the one of the site                                         *)
(* ======================================================================== *)

Lemma binds_simple_plain (s : simple) (y : variable) :
  is_plain s = true -> binds_simple s y = false.
Proof. destruct s; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma simple_among_plain (s : simple) (params : list variable) :
  is_plain s = true -> simple_among s params = false.
Proof. destruct s; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma apply_cont_gives_plain (k : continuation) (i : nat) (s : simple)
    (ac : apply_cont) :
  apply_cont_gives k i s ac = apply_cont_gives_simple k i s true ac.
Proof. reflexivity. Qed.

Fixpoint all_give_plain (k : continuation) (i : nat) (s : simple)
    (Hplain : is_plain s = true) (e : expr) {struct e} :
  all_give k i s e = all_give_simple k i s true e
with handler_all_give_plain (k : continuation) (i : nat) (s : simple)
    (Hplain : is_plain s = true) (h : continuation_handler) {struct h} :
  handler_all_give k i s h = handler_all_give_simple k i s true h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [all_give all_give_simple]; try reflexivity.
    + rewrite (binds_simple_plain s y Hplain). apply all_give_plain, Hplain.
    + apply all_give_plain, Hplain.
    + rewrite (handler_all_give_plain k i s Hplain),
        (all_give_plain k i s Hplain).
      reflexivity.
    + rewrite (all_give_plain k i s Hplain). f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [forallb]. rewrite (handler_all_give_plain k i s Hplain), IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_all_give handler_all_give_simple].
    rewrite (simple_among_plain s params Hplain).
    apply all_give_plain, Hplain.
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

Lemma bind_const_param_equivalent (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  bind_const_param i k h body = Some (h', body') ->
  expr_equivalent (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold bind_const_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ] eqn:Hkind;
    [ | discriminate ].
  destruct (first_given k i body) as [[s | ] | ]; try discriminate.
  destruct
    (is_plain s && negb (hm_is_exn_handler metadata) && all_give k i s body
     && negb (among p (drop_nth i params)))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond Hfresh].
  apply andb_true_iff in Hcond as [Hcond Hgive].
  apply andb_true_iff in Hcond as [Hplain _].
  apply negb_true_iff in Hfresh.
  rewrite (all_give_plain k i s Hplain) in Hgive.
  exact (given_param_equivalent i k s p kind params metadata handler body
    Hparam Hkind (simple_among_plain s params Hplain) Hfresh Hgive).
Qed.

Lemma bind_const_from_equivalent (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    expr_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (bind_const_from i k h body))
         (snd (bind_const_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [bind_const_from].
  - intros vars conts statics hp rd. apply equivalent_refl.
  - destruct (bind_const_param i k h body) as [[h' body'] | ] eqn:Hparam;
      [ | apply IH ].
    exact (expr_equivalent_trans _ _ _
      (bind_const_param_equivalent i k h body h' body' Hparam)
      (IH k h' body')).
Qed.

(* Each layer is a context around what the induction gives; then the
   parameters of the [Let_cont] are looked at. *)
Fixpoint bind_const_params_equivalent_aux (e : expr) :
  expr_equivalent e (bind_const_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [bind_const_params];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (bind_const_params_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (bind_const_params_equivalent_aux body)).
  - assert (Hlayers :
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (bind_const_params handler))
           (bind_const_params body))).
    { apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (bind_const_params handler)) body)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_handler k params metadata Hole body) _ _
          (not_many _) (bind_const_params_equivalent_aux handler)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (bind_const_params handler)) Hole)
          _ _ (not_many _) (bind_const_params_equivalent_aux body)). }
    pose proof
      (bind_const_from_equivalent (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as Hfrom.
    destruct
      (bind_const_from (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as [h' body'].
    exact (expr_equivalent_trans _ _ _ Hlayers Hfrom).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (bind_const_params_equivalent_aux body)).
Qed.

Theorem bind_const_params_equivalent : bind_const_params_equivalent_statement.
Proof. exact bind_const_params_equivalent_aux. Qed.
