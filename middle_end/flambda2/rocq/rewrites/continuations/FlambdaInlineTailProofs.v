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

(* Proofs of the statement of FlambdaInlineTailSpec.v. Nothing here needs
   review.

   The proof is that of FlambdaNormaliseProofs.inline_tail_conts_refines
   with equivalences in place of refinements: each layer is a context
   around what the induction gives (FlambdaContextProofs.plug_equivalent),
   and where the body ends by applying the continuation, the application
   and the [Let]s of the parameters run to the handler under variables
   that agree on its free variables, or both go wrong: where an argument
   has no value (FlambdaApplyProofs.let_params_wrong), and where the
   values are not of the kinds of the handler
   (FlambdaApplyProofs.let_params_wrong_kinds), which is where the
   handler must give one kind for each parameter. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaApplySpec FlambdaApplyProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaInlineTailSpec.
Import ListNotations.

Lemma expr_equivalent_refl (e : expr) : expr_equivalent e e.
Proof. intros vars conts statics hp rd. apply equivalent_refl. Qed.

(* The body under the binding of [k], against the body with the handler
   in place: the [Let]s run alike; then the one applies [k] and the other
   binds the parameters and goes on, to the handler under environments
   that agree on its free variables. *)
Lemma inline_cont_at_tail_equivalent (k : continuation)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (body : expr) :
  length params = length (hm_params_kinds metadata) ->
  ∀ inlined : expr,
    inline_cont_at_tail k params metadata handler body = Some inlined ->
    ∀ (vars def_vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record),
      (∀ x : variable,
        var_free_in_handler x (Handler params metadata handler) ->
        lookup x vars = lookup x def_vars) ->
      equivalent
        (State body vars
           ((k, Closure (Handler params metadata handler) def_vars conts [])
            :: conts)
           statics hp rd)
        (State inlined vars conts statics hp rd).
Proof.
  intros Hkinds.
  induction body
    as [ x kind d body IH | bindings body IH | k' h body IH | handlers body IH
       | ap | ac | scrutinee arms | ];
    intros inlined Heq vars def_vars conts statics hp rd Hagree; simpl in Heq;
    try discriminate.
  - (* Let *)
    match type of Heq with
    | context [if ?free then _ else _] => destruct free eqn:Hfree
    end;
      [ discriminate | ].
    change (var_free_in_handler_b x (Handler params metadata handler) = false)
      in Hfree.
    destruct (inline_cont_at_tail k params metadata handler body)
      as [body' | ];
      [ | discriminate ].
    simpl in Heq. injection Heq as <-.
    apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply (IH body' eq_refl). intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ | exact (Hagree y Hy) ].
    apply var_free_in_handler_b_sound in Hy. congruence.
  - (* Let_static *)
    destruct (inline_cont_at_tail k params metadata handler body)
      as [body' | ];
      [ | discriminate ].
    simpl in Heq. injection Heq as <-.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    exact (IH body' eq_refl vars def_vars conts extended hp rd_ Hagree).
  - (* Apply_cont *)
    destruct
      (Pos.eqb (ac_continuation ac) k
       && no_trap_action ac
       && Nat.eqb (length params) (length (ac_args ac))
       && forallb (fun param => negb (var_in_simples param (ac_args ac)))
            params)
      eqn:Hcheck; [ | discriminate ].
    injection Heq as <-.
    apply andb_true_iff in Hcheck as [Hcheck Hcapture].
    apply andb_true_iff in Hcheck as [Hcheck Hlength].
    apply andb_true_iff in Hcheck as [Hk _]. apply Pos.eqb_eq in Hk.
    apply Nat.eqb_eq in Hlength.
    assert (Hfresh :
      ∀ param : variable,
        param ∈ params -> ~ var_free_in_simples param (ac_args ac)).
    { intros param Hin Hfree. apply var_in_simples_sound in Hfree.
      rewrite forallb_forall in Hcapture. specialize (Hcapture param Hin).
      apply negb_true_iff in Hcapture. congruence. }
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
    + destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Htested.
      * assert (Hvs : length params = length vs)
          by (rewrite (eval_simples_length vars (ac_args ac) vs Hargs);
              exact Hlength).
        destruct (bind_params_some params vs def_vars Hvs)
          as [handler_vars Hbind].
        destruct (bind_params_some params vs vars Hvs) as [bound Hbound].
        destruct
          (let_params_run params (hm_params_kinds metadata) (ac_args ac) vs
             vars bound handler conts statics hp rd Hargs
             (proj1 (kinds_eqb_eq _ _) Htested) Hfresh Hbound)
          as [n Hrun].
        apply
          (equivalent_trans _ (State handler handler_vars conts statics hp rd)).
        { apply step_equivalent.
          apply
            (step_apply_cont_handler _ _ _ _ _ rd_ vs params metadata handler
            def_vars conts []);
            [ exact Hargs | rewrite Hk; apply resolve_head | exact Htested
            | exact Hbind ]. }
        apply (equivalent_trans _ (State handler bound conts statics hp rd)).
        { apply coincide_equivalent.
          - intros x Hx.
            apply (bind_params_agree params vs def_vars vars handler_vars
              bound Hbind Hbound).
            destruct (in_dec Pos.eq_dec x params) as [Hin | Hnotin];
              [ left; exact Hin | right ].
            apply Hagree. split; assumption.
          - intros k' _. apply (conts_agree_same identity_renaming).
            reflexivity. }
        apply equivalent_sym, (equivalent_of_run n 0). exact Hrun.
      * apply both_wrong_equivalent.
        -- apply step_undefined_wrong. rewrite step_apply_cont, Hargs.
           unfold apply_continuation. rewrite Hk, resolve_head. simpl.
           rewrite Htested. reflexivity.
        -- apply
             (let_params_wrong_kinds params (hm_params_kinds metadata)
                (ac_args ac) vs vars handler conts statics _ rd_ Hlength Hkinds
                Hargs);
             [ | exact Hfresh ].
           intros Heq. rewrite Heq, kinds_eqb_refl in Htested.
           discriminate Htested.
    + apply both_wrong_equivalent.
      * apply step_undefined_wrong. rewrite step_apply_cont, Hargs.
        reflexivity.
      * exact
          (let_params_wrong params (hm_params_kinds metadata) (ac_args ac)
             vars handler conts statics _ rd_ Hlength Hargs Hfresh).
Qed.

(* Each layer is a context around what the induction gives; then the
   handler goes into the body where that ends by applying it. *)
Fixpoint inline_tail_conts_kinded_equivalent_aux (e : expr) :
  expr_equivalent e (inline_tail_conts_kinded e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try apply expr_equivalent_refl.
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (inline_tail_conts_kinded_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (inline_tail_conts_kinded_equivalent_aux body)).
  - assert (Hlayers :
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (inline_tail_conts_kinded handler))
           (inline_tail_conts_kinded body))).
    { apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (inline_tail_conts_kinded handler))
           body)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_handler k params metadata Hole body)
          _ _ (not_many _)
          (inline_tail_conts_kinded_equivalent_aux handler)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (inline_tail_conts_kinded handler))
             Hole)
          _ _ (not_many _) (inline_tail_conts_kinded_equivalent_aux body)). }
    destruct (Nat.eqb (length params) (length (hm_params_kinds metadata)))
      eqn:Hkinds; [ | exact Hlayers ].
    apply Nat.eqb_eq in Hkinds.
    destruct
      (inline_cont_at_tail k params metadata
         (inline_tail_conts_kinded handler) (inline_tail_conts_kinded body))
      as [inlined | ] eqn:Hinline; [ | exact Hlayers ].
    apply (expr_equivalent_trans _ _ _ Hlayers). intros vars conts statics hp
      rd.
    eapply equivalent_trans;
      [ apply step_equivalent, step_let_cont_non_recursive | ].
    apply (inline_cont_at_tail_equivalent k params metadata _ _ Hkinds inlined
      Hinline).
    intros x _. reflexivity.
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (inline_tail_conts_kinded_equivalent_aux body)).
Qed.

Theorem inline_tail_conts_kinded_equivalent :
  inline_tail_conts_kinded_equivalent_statement.
Proof. exact inline_tail_conts_kinded_equivalent_aux. Qed.
