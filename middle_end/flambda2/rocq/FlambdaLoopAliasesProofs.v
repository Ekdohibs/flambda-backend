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

(* Proofs of the statements of FlambdaLoopAliasesSpec.v. Nothing here
   needs review.

   The induction is that of FlambdaAliasesProofs.v, under
   a typing environment that the variables satisfy. At a recursive
   [Let_cont] the handlers are replaced all at once
   (FlambdaContextProofs.let_cont_recursive_handlers_refine_where): each
   new handler is equivalent to the old one under the variables of the
   closure with the parameters bound, as those satisfy what was known
   less what mentions the parameters. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaRecHandlersProofs FlambdaLoopAliasesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaTypesProofs
  FlambdaAliasesSpec FlambdaAliasesProofs.
Import ListNotations.

Fixpoint follow_aliases_in_loops_binds_statics (e : expr) :
  ∀ tenv : typing_env,
    binds_statics (follow_aliases_in_loops tenv e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros tenv; simpl; try reflexivity.
  - apply follow_aliases_in_loops_binds_statics.
  - rewrite follow_aliases_in_loops_binds_statics. reflexivity.
  - rewrite !follow_aliases_in_loops_binds_statics. reflexivity.
  - rewrite follow_aliases_in_loops_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite rewrite_code_body_binds_statics,
      follow_aliases_in_loops_binds_statics, IH.
    reflexivity.
Qed.

Lemma rewrite_code_body_equivalent_in (tenv : typing_env) (f : expr -> expr)
    (body : expr) :
  (∀ e : expr, expr_equivalent_in tenv e (f e)) ->
  expr_equivalent_in tenv body (rewrite_code_body f body).
Proof.
  intros Hf. unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)));
    [ apply Hf | intros vars conts statics hp rd _; apply equivalent_refl ].
Qed.

(* Handlers that are equivalent, one with one, under what is known less
   what mentions their parameters. *)
Definition handlers_equivalent_in (tenv : typing_env)
    (handlers handlers' : list (continuation * continuation_handler)) : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh'
       ∧ ∃ (params : list variable) (metadata : handler_metadata)
           (handler handler' : expr),
           snd kh = Handler params metadata handler
           ∧ snd kh' = Handler params metadata handler'
           ∧ expr_equivalent_in (drop_vars params tenv) handler handler')
    handlers handlers'.

(* Such handlers refine each other where they run, in the four forms
   that the replacement of the handlers of a recursive [Let_cont] asks
   for. *)
Lemma handlers_equivalent_related (tenv : typing_env) (vars : env)
    (handlers handlers' : list (continuation * continuation_handler)) :
  env_satisfies tenv vars ->
  handlers_equivalent_in tenv handlers handlers' ->
  handlers_refine always vars handlers handlers'
  ∧ handlers_refine always vars handlers' handlers.
Proof.
  intros Hsat Hall. unfold handlers_refine.
  induction Hall
    as [ | [k h] [k' h'] handlers handlers'
           (Hk & params & metadata & handler & handler' & Hh & Hh' & Hequiv) _
           [[Hfwd Hbwd] [Hfwd' Hbwd']]].
  - repeat split; apply Forall2_nil.
  - simpl in Hk, Hh, Hh'. subst k' h h'.
    assert (Hruns :
      ∀ (vs : list value) (handler_vars : env),
        bind_params params vs vars = Some handler_vars ->
        env_satisfies (drop_vars params tenv) handler_vars).
    { intros vs handler_vars Hbind.
      apply (env_satisfies_bind_params _ params vs vars handler_vars);
        [ | intros param Hin; apply drop_vars_fresh, Hin | exact Hbind ].
      exact (env_satisfies_incl tenv _ vars (drop_vars_incl params tenv)
        Hsat). }
    repeat split; (apply Forall2_cons; [ | assumption ]);
      (split; [ reflexivity | ]); apply Handler_related;
      intros vs handler_vars conts statics hp rd _ _ Hbind;
      apply equivalent_refines;
      first
        [ apply Hequiv, (Hruns vs handler_vars Hbind)
        | apply equivalent_sym, Hequiv, (Hruns vs handler_vars Hbind) ].
Qed.

Lemma let_cont_recursive_handlers_equivalent_in (tenv : typing_env)
    (handlers handlers' : list (continuation * continuation_handler))
    (body : expr) :
  handlers_equivalent_in tenv handlers handlers' ->
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  expr_equivalent_in tenv (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body).
Proof.
  intros Hall Hbinds vars conts statics hp rd Hsat.
  destruct (handlers_equivalent_related tenv vars handlers handlers' Hsat Hall)
    as [[Hfwd Hbwd] [Hfwd' Hbwd']].
  apply refines_antisymmetric;
    apply (let_cont_recursive_handlers_refine_where always always_extend);
    first [ exact I | assumption | symmetry; assumption ].
Qed.

Fixpoint follow_aliases_in_loops_equivalent_in (e : expr) :
  ∀ tenv : typing_env,
    expr_equivalent_in tenv e (follow_aliases_in_loops tenv e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros tenv; simpl;
    try (intros vars conts statics hp rd _; apply equivalent_refl).
  - apply let_equivalent_in, follow_aliases_in_loops_equivalent_in.
  - apply let_static_equivalent_in, follow_aliases_in_loops_equivalent_in.
  - apply (expr_equivalent_in_trans tenv _
      (Let_cont_non_recursive k
         (Handler params metadata
            (follow_aliases_in_loops (drop_vars params tenv) handler))
         body)).
    + apply let_cont_handler_equivalent_in,
        follow_aliases_in_loops_equivalent_in.
    + intros vars conts statics hp rd Hsat.
      apply let_cont_non_recursive_equivalent_step.
      apply (follow_aliases_in_loops_equivalent_in body tenv), Hsat.
  - apply (expr_equivalent_in_trans tenv _
      (Let_cont_recursive
         (map (fun '(k, h) => (k, follow_aliases_in_loop tenv h)) handlers)
         body)).
    + apply let_cont_recursive_handlers_equivalent_in.
      * induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ apply Forall2_nil | ].
        apply Forall2_cons; [ | exact IH ].
        split; [ reflexivity | ]. simpl.
        exists params, metadata, handler,
          (rewrite_code_body (follow_aliases (drop_vars params tenv))
             (follow_aliases_in_loops (drop_vars params tenv) handler)).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        apply (expr_equivalent_in_trans _ _
          (follow_aliases_in_loops (drop_vars params tenv) handler));
          [ apply follow_aliases_in_loops_equivalent_in | ].
        apply rewrite_code_body_equivalent_in. intros e.
        apply follow_aliases_equivalent_in.
      * unfold handlers_bind_statics.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite rewrite_code_body_binds_statics,
          follow_aliases_in_loops_binds_statics, IH.
        reflexivity.
    + intros vars conts statics hp rd Hsat.
      apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply (follow_aliases_in_loops_equivalent_in body tenv), Hsat.
Qed.

Theorem follow_aliases_in_loops_equivalent :
  follow_aliases_in_loops_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (follow_aliases_in_loops_equivalent_in e []). intros x t [].
Qed.
