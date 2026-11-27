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

(* That the rebuild with changes of calling convention of the reaper
   commutes with the erasure of regions on the terms of
   FlambdaReaperRegionsSpec.regions_plain whose code that changes returns
   no region, for any facts and solutions that agree on what the rebuild
   reads, the same table of the code that cannot change, and code deps
   whose nodes are not the constants. It is the commutation of
   FlambdaReaperRegionsRebuildProofs.v, case for case, with the calls and
   the code of the rebuild with changes of calling convention.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsCcProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaRegionsSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsRebuildProofs
  FlambdaReaperRegionsTraverseProofs FlambdaReaperRegionsKeepsProofs.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaReaperFlowProofs.
Import ListNotations.

(* ======================================================================== *)
(* The metadata of the code of the erased term                              *)
(* ======================================================================== *)

Lemma flat_map_map {A B C : Type} (f : B -> list C) (g : A -> B)
    (l : list A) :
  flat_map f (map g l) = flat_map (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

(* The erasure keeps the metadata of code. *)
Lemma bound_code_metadata_erase :
  (∀ e : expr, ∀ c,
     bound_code_metadata (erase c e) = bound_code_metadata e)
  ∧ (∀ h : continuation_handler, ∀ c,
       handler_bound_code_metadata (erase_handler c h)
       = handler_bound_code_metadata h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [bound_code_metadata]; apply IH).
    destruct p as [ | | | | r]; try (cbn [bound_code_metadata]; apply IH).
    destruct (erase_region_primitive x r args);
      cbn [bound_code_metadata]; apply IH.
  - intros bindings body _ IH c. cbn [erase bound_code_metadata].
    rewrite IH, flat_map_map. f_equal. apply flat_map_ext_in.
    intros [id cd | | ] _; [ destruct cd | | ]; reflexivity.
  - intros k h body IHh IH c. cbn [erase bound_code_metadata].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase bound_code_metadata].
    rewrite IH, flat_map_map. f_equal. apply flat_map_ext_in.
    intros [k h] Hin. exact (IHh k h Hin _).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c. cbn [erase_handler
      handler_bound_code_metadata]. apply IH.
Qed.

(* The code that the rebuild makes is erased in its body alone. *)
Lemma erase_cc_code_of (types : subkinds_rewriting) (id : code_id)
    (change : calling_convention_change) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  erase_code (cc_code_of types id change params mc ret exn md body)
  = cc_code_of types id change params mc ret exn md (erase [] body).
Proof. destruct change; reflexivity. Qed.

(* The places of the regions of a handler that the rebuild makes. *)
Lemma rebuilt_handler_places_cc (F0 : facts) (s0 : solution)
    (cc0 : node -> bool) (cds0 : finite_map code_dep) (b0 : bool)
    (u0 : list code_id) (codes0 : finite_map code_metadata) (f0 : positive)
    (en : rebuild_env) (dc : bool) (mask : list bool)
    (h : continuation_handler) :
  forallb kind_not_region (handler_kinds h) = true ->
  forallb negb
    (region_params
       (rebuild_handler_cc F0 s0 cc0 cds0 b0 u0 codes0 f0 None en dc mask h))
  = true.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite rebuild_handler_cc_unfold. cbn [region_params handler_kinds].
  apply region_places_plain, kept_metadata_plain, H.
Qed.

(* ======================================================================== *)
(* The parts of the two rebuilds                                            *)
(* ======================================================================== *)

Section CommuteCc.

(* As in FlambdaReaperRegionsRebuildProofs.v, and the table of the code
   that cannot change, the code deps and the metadata of the code of the
   unit, the same for both. *)
Variables F F' : facts.
Variables sol sol' : solution.
Variable b : bool.
Variable ucids : list code_id.
Variable ff : positive.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable codes : finite_map code_metadata.

Hypothesis HC : f_constructor F' = f_constructor F.
Hypothesis HA : f_accessor F' = f_accessor F.
Hypothesis HP : f_parameter F' = f_parameter F.
Hypothesis Hpac :
  ∀ (x : node) (i : nat) (y : node),
    (x, i, y) ∈ f_parameter F -> y <> Node_all_constants.
Hypothesis Hu :
  ∀ n : node,
    n <> Node_all_constants -> s_has_usage sol' n = s_has_usage sol n.
Hypothesis Hsrc : s_has_source sol' = s_has_source sol.
Hypothesis Has : s_any_source sol' = s_any_source sol.
Hypothesis Hss : s_sources sol' = s_sources sol.
Hypothesis Hfu : s_field_used sol' = s_field_used sol.

(* The nodes of the parameters and the results of code are not the
   constants. *)
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep),
    lookup id cds = Some cd ->
    ∀ n : node, n ∈ cd_params cd ++ cd_return cd -> n <> Node_all_constants.

(* The code that a call is known to call, if it changes its calling
   convention, returns no region. *)
Hypothesis Hres :
  ∀ (ap : apply_expr) (pd rd : list param_decision),
    apply_calling_convention_change F sol cc cds ucids codes ap
    = Changing_calling_convention pd rd ->
    forallb kind_not_region (kept_kinds rd) = true.

Lemma dead_eq (x : variable) (kind : flambda_kind) :
  is_dead_var sol' x kind = is_dead_var sol x kind.
Proof. apply dead_var_eq, Hsrc. Qed.

Lemma used_eq (x : variable) (kind : flambda_kind) :
  is_var_used sol' x kind = is_var_used sol x kind.
Proof. apply var_used_eq, Hu. Qed.

Lemma decisions_of_eq (nodes : list node) (kinds : list flambda_kind) :
  (∀ n : node, n ∈ nodes -> n <> Node_all_constants) ->
  decisions_of sol' nodes kinds = decisions_of sol nodes kinds.
Proof.
  intros Hn. unfold decisions_of. apply map_ext_in. intros [n k] Hin.
  unfold is_node_used.
  destruct k; try reflexivity;
    rewrite (Hu n (Hn n (in_combine_l _ _ _ _ Hin))); reflexivity.
Qed.

Lemma code_change_eq (id : code_id) (md : code_metadata) :
  code_calling_convention_change sol' cc cds ucids id md
  = code_calling_convention_change sol cc cds ucids id md.
Proof.
  unfold code_calling_convention_change.
  destruct (cannot_change_calling_convention cc ucids id); [ reflexivity | ].
  destruct (lookup id cds) as [cd | ] eqn:Hl; [ | reflexivity ].
  rewrite (decisions_of_eq (cd_params cd)), (decisions_of_eq (cd_return cd));
    [ reflexivity | | ]; intros n Hn; apply (Hcds id cd Hl);
    apply in_or_app; [ right | left ]; exact Hn.
Qed.

Lemma apply_change_eq (ap : apply_expr) :
  apply_calling_convention_change F' sol' cc cds ucids codes ap
  = apply_calling_convention_change F sol cc cds ucids codes ap.
Proof.
  unfold apply_calling_convention_change, code_id_actually_called.
  rewrite (code_id_actually_directly_called_eq F F' sol sol' HC HA Hsrc Has
             Hss).
  destruct (match ap_call_kind ap with
            | Indirect_unknown_arity => None
            | _ => _
            end) as [id | ]; [ | reflexivity ].
  unfold get_calling_convention_change.
  destruct (lookup id codes); [ apply code_change_eq | reflexivity ].
Qed.

Section Env.

Variables en en' : rebuild_env.
Hypothesis Hen : env_rel en en'.

Lemma get_args_eq (ds : list param_decision) (args : list simple) :
  forallb (simple_not_region (env_kinds en)) args = true ->
  get_args_with_kinds sol' en' ds args = get_args_with_kinds sol en ds args.
Proof.
  intros H. unfold get_args_with_kinds. apply flat_map_ext_in.
  intros [a d] Hin. destruct d; [ | reflexivity ].
  rewrite forallb_forall in H.
  rewrite (rewrite_simple_eq sol sol' Hu en en' Hen a
             (H a (in_combine_l _ _ _ _ Hin))).
  reflexivity.
Qed.

Lemma groups_eq (a : arity) (args : list simple) (pd : list param_decision) :
  forallb (simple_not_region (env_kinds en)) args = true ->
  map (fun '(args, decisions) => get_args_with_kinds sol' en' decisions args)
    (combine (group_by_parameter a args) (group_by_parameter a pd))
  = map (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
      (combine (group_by_parameter a args) (group_by_parameter a pd)).
Proof.
  intros H. apply map_ext_in. intros [g ds] Hin. apply get_args_eq.
  apply in_combine_l in Hin. rewrite forallb_forall in H |- *.
  intros x Hx. apply H. unfold group_by_parameter in Hin.
  exact (group_by_sizes_in _ _ _ _ Hin Hx).
Qed.

Lemma wrapper_cc_eq (ap : apply_expr)
    (fd : list (param_decision * flambda_kind)) :
  make_apply_wrapper_cc ff en' ap fd = make_apply_wrapper_cc ff en ap fd.
Proof.
  destruct Hen as (Hc & _ & _). unfold make_apply_wrapper_cc, apply_decisions.
  rewrite Hc. reflexivity.
Qed.

Lemma rebuild_apply_cc_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en))
    (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  rebuild_apply_cc F' sol' cc cds b ucids codes ff None en' ap
  = rebuild_apply_cc F sol cc cds b ucids codes ff None en ap.
Proof.
  intros H. pose proof H as H0. cbn [forallb] in H.
  apply andb_true_iff in H as [Hc H].
  rewrite forallb_app in H. apply andb_true_iff in H as [Hargs Hextra].
  unfold rebuild_apply_cc. rewrite apply_change_eq.
  destruct (apply_calling_convention_change F sol cc cds ucids codes ap)
    as [ | pd rd].
  - exact (rebuild_apply_eq F F' sol sol' b ucids ff HC HA HP Hpac Hu Hsrc
             Has Hss en en' Hen ap H0).
  - rewrite (simple_is_dead_eq sol sol' Hsrc en en' Hen _ Hc).
    rewrite (existsb_ext_in (simple_is_dead sol' en') (simple_is_dead sol en)).
    + destruct (_ || _); [ reflexivity | ].
      rewrite (groups_eq _ _ _ Hargs),
        (rewrite_call_kind_eq F F' sol sol' b ucids HC HA Hsrc Has Hss),
        (rebuild_exn_extra_args_eq sol sol' Hu en en' Hen ap Hextra),
        wrapper_cc_eq.
      reflexivity.
    + intros s Hs. rewrite forallb_forall in Hargs.
      exact (simple_is_dead_eq sol sol' Hsrc en en' Hen s (Hargs s Hs)).
Qed.

End Env.

(* What the rebuild makes of a call is erased to itself. *)
Lemma erase_rebuild_apply_cc (en : rebuild_env) (ap : apply_expr) :
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  erase [] (rebuild_apply_cc F sol cc cds b ucids codes ff None en ap)
  = rebuild_apply_cc F sol cc cds b ucids codes ff None en ap.
Proof.
  intros Hr. unfold rebuild_apply_cc.
  destruct (apply_calling_convention_change F sol cc cds ucids codes ap)
    as [ | pd rd] eqn:Hch; [ apply erase_rebuild_apply, Hr | ].
  destruct (_ || _); [ reflexivity | ].
  unfold make_apply_wrapper_cc.
  match goal with
  | |- erase [] (if ?c then _ else _) = _ => destruct c
  end; [ reflexivity | ].
  cbn [erase]. rewrite erase_handler_no_region; [ reflexivity | ].
  cbn [hm_params_kinds make_handler_metadata].
  apply kept_kinds_combine. exact (Hres ap pd rd Hch).
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Abbreviation RB := (rebuild_expr_cc F sol cc cds b ucids codes ff None).
Abbreviation RB' := (rebuild_expr_cc F' sol' cc cds b ucids codes ff None).
Abbreviation RH := (rebuild_handler_cc F sol cc cds b ucids codes ff None).
Abbreviation RH' := (rebuild_handler_cc F' sol' cc cds b ucids codes ff None).

Lemma rebuilt_conts_plain_cc (en : rebuild_env)
    (hs : list (continuation * continuation_handler)) :
  (∀ k h, (k, h) ∈ hs -> forallb kind_not_region (handler_kinds h) = true) ->
  conts_plain
    (map (fun '(k, h) => (k, region_params h))
       (map (fun '(k, h) =>
               (k, RH en false (handler_mask sol false h) h)) hs) ++ []).
Proof.
  intros H. apply conts_plain_app; [ | apply conts_plain_nil ].
  intros k p Hin. apply in_map_iff in Hin as [[k' h'] [Heq Hin]].
  injection Heq as <- <-.
  apply in_map_iff in Hin as [[k0 h0] [Heq Hin]].
  injection Heq as <- <-. apply rebuilt_handler_places_cc.
  exact (H k0 h0 Hin).
Qed.

Lemma rebuild_cc_erase_commutes_gen :
  (∀ e : expr, ∀ en en' : rebuild_env,
     regions_plain (env_kinds en) e = true -> env_rel en en' ->
     pending_ok en e ->
     erase [] (RB en e) = RB' en' (erase [] e))
  ∧ (∀ h : continuation_handler,
       ∀ (en en' : rebuild_env) (dc : bool) (mask : list bool),
         handler_regions_plain (env_kinds en) h = true -> env_rel en en' ->
         erase_handler [] (RH en dc mask h)
         = RH' en' dc mask (erase_handler [] h)).
Proof.
  apply regions_ind.
  - (* Let *)
    intros x kind d body IH IH2 en en' Hpl Hen Hpend.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      assert (Hnil : env_projections en = [])
        by (apply (pending_ok_let_nil en x kind
                     (Prim (Region_primitive p) args) body);
            [ intros ? ? ? ?; discriminate | exact Hpend ]).
      rewrite (rebuild_cc_let_other F sol cc cds b ucids codes ff None en x
                 kind _ body (pending_prim _ _ _ _) eq_refl).
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl.
      * (* Begin_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        cbn [is_dead_var is_var_used rebuild_named map erase
               erase_region_primitive].
        apply IH; [ exact Hb | apply env_rel_bind_region; assumption
                  | apply pending_ok_bind ].
      * (* Begin_try_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        cbn [is_dead_var is_var_used rebuild_named map erase
               erase_region_primitive].
        apply IH; [ exact Hb | apply env_rel_bind_region; assumption
                  | apply pending_ok_bind ].
      * (* End_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        cbn [erase erase_region_primitive].
        rewrite (rebuild_cc_let_other F' sol' cc cds b ucids codes ff None en'
                   x kind (Simple unit_simple) _ (pending_simple _ _ _)
                   eq_refl).
        rewrite dead_eq, used_eq.
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- cbn [rebuild_named map erase erase_region_primitive rewrite_simple
                  unit_simple].
           f_equal. apply IH; [ exact Hb | apply env_rel_bind, Hen
                              | apply pending_ok_bind ].
        -- apply IH; [ exact Hb | apply env_rel_bind, Hen
                     | apply pending_ok_bind ].
      * (* End_try_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        cbn [erase erase_region_primitive].
        rewrite (rebuild_cc_let_other F' sol' cc cds b ucids codes ff None en'
                   x kind (Prim (Unary (Opaque_identity Value)) [unit_simple])
                   _ (pending_prim _ _ _ _) eq_refl).
        rewrite dead_eq, used_eq.
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- cbn [rebuild_named map erase erase_region_primitive rewrite_simple
                  unit_simple].
           f_equal. apply IH; [ exact Hb | apply env_rel_bind, Hen
                              | apply pending_ok_bind ].
        -- apply IH; [ exact Hb | apply env_rel_bind, Hen
                     | apply pending_ok_bind ].
    + (* Not a primitive of regions. *)
      rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [Hpl Hla].
      apply andb_true_iff in Hpl as [Hpl Hset].
      apply andb_true_iff in Hpl as [Hk Hd].
      destruct Hen as (Hc & Hps & Hkinds).
      assert (Hen : env_rel en en') by (split; [ | split ]; assumption).
      rewrite (erase_let_plain _ _ _ _ _ Hrp).
      unfold la_check in Hla.
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
      * (* A local allocation. *)
        apply la_body_some in Hlab. subst body.
        apply andb_true_iff in Hla as [Hla Hb'].
        apply andb_true_iff in Hla as [Hla Hargs].
        apply andb_true_iff in Hla as [Hla Hk'].
        apply andb_true_iff in Hla as [Halloc Hy].
        apply Pos.eqb_eq in Hy. subst y.
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
        destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
        apply Pos.eqb_eq in Hargs. subst x'.
        cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
        pose proof (allocation_not_projection d Halloc) as Hnproj.
        (* The two [Let]s of [x] are kept or removed together. *)
        assert (Hinner :
          ∀ en1 : rebuild_env,
            env_kinds en1 = (x, kind) :: env_kinds en ->
            env_projections en1 = [] ->
            env_conts en1 = env_conts en ->
            is_dead_var sol x kind = false ->
            erase []
              (RB en1
                 (Let x kind
                    (Prim (Region_primitive Local_allocation) (Var x :: rs))
                    body'))
            = RB' (bind_variables en' [(x, kind)]) (erase [] body')).
        { intros en1 Hk1 Hp1 Hc1 Hnd.
          rewrite (rebuild_cc_let_other F sol cc cds b ucids codes ff None en1
                     x kind
                     (Prim (Region_primitive Local_allocation) (Var x :: rs))
                     body' (pending_prim _ _ _ _) eq_refl).
          rewrite Hnd.
          assert (Hrel : env_rel (bind_variables en1 [(x, kind)])
                           (bind_variables en' [(x, kind)])).
          { split; [ cbn; rewrite Hc1, Hc; reflexivity | ].
            split; [ reflexivity | ].
            intros z Hz. unfold region_variable in Hz.
            cbn [bind_variables bind_variables_then env_kinds rev app lookup]
              in Hz |- *.
            rewrite Hk1 in Hz |- *. cbn [lookup] in Hz |- *.
            destruct (Pos.eqb z x); [ reflexivity | exact (Hkinds z Hz) ]. }
          assert (Hb'' : regions_plain
                           (env_kinds (bind_variables en1 [(x, kind)]))
                           body' = true)
            by (cbn [bind_variables bind_variables_then env_kinds rev app];
                rewrite Hk1; exact Hb').
          destruct (is_var_used sol x kind) eqn:Hu1.
          - cbn [rebuild_named map erase erase_region_primitive].
            assert (Hrw : rewrite_simple sol en1 (Var x) = Var x).
            { cbn [rewrite_simple]. unfold kind_of_variable. rewrite Hk1.
              cbn [lookup]. rewrite Pos.eqb_refl, Hu1. reflexivity. }
            rewrite Hrw. cbn [erase erase_region_primitive].
            rewrite Pos.eqb_refl.
            apply IH2 with (y := x) (kind' := kind)
              (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
              [ reflexivity | exact Hb'' | exact Hrel | apply pending_ok_bind ].
          - apply IH2 with (y := x) (kind' := kind)
              (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
              [ reflexivity | exact Hb'' | exact Hrel
              | apply pending_ok_bind ]. }
        destruct (named_is_set d) eqn:Hs.
        -- rewrite (rebuild_cc_let_set F sol cc cds b ucids codes ff None en x
                      kind d _ (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite (rebuild_cc_let_set F' sol' cc cds b ucids codes ff None en'
                      x kind d _ (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite !(set_plain_region_head _ _ _ _ _ _ _ Hset).
           cbn [existsb]. rewrite !orb_false_r, dead_eq, used_eq.
           destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
           destruct (is_var_used sol x kind) eqn:Hused.
           ++ rewrite (erase_let_plain _ _ _ _ _
                         (rebuild_set_plain _ _ _ _ Hrp)).
              rewrite (rebuild_set_eq sol sol' Hu Hfu _ _ Hen _ _ Hd).
              unfold set_closure_nodes. rewrite HC. f_equal.
              apply (Hinner (bind_variables_then en [(x, kind)] []));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
           ++ apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
        -- rewrite (rebuild_cc_let_other F sol cc cds b ucids codes ff None en x
                      kind d _ (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite (rebuild_cc_let_other F' sol' cc cds b ucids codes ff None
                      en' x kind d _ (pending_not_projection _ _ _ Hnproj) Hs).
           rewrite dead_eq, used_eq.
           destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
           destruct (is_var_used sol x kind) eqn:Hused.
           ++ rewrite (erase_let_plain _ _ _ _ _
                         (rebuild_named_plain _ _ _ _ Hrp)).
              rewrite (rebuild_named_eq sol sol' Hu Hfu _ _ Hen _ _ Hd).
              f_equal.
              apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
           ++ apply (Hinner (bind_variables en [(x, kind)]));
                [ reflexivity | reflexivity | reflexivity | reflexivity ].
      * (* No local allocation follows. *)
        destruct (is_pending_projection en x d) eqn:Hp.
        -- assert (Hp' : is_pending_projection en' x d = true)
             by (unfold is_pending_projection in *; rewrite Hps; exact Hp).
           rewrite (rebuild_cc_let_pending F sol cc cds b ucids codes ff None
                      en x kind d body Hp).
           rewrite (rebuild_cc_let_pending F' sol' cc cds b ucids codes ff None
                      en' x kind d (erase [] body) Hp').
           rewrite (erase_let_plain _ _ _ _ _ Hrp). f_equal.
           rewrite Hps. apply IH;
             [ exact Hla | apply env_rel_bind_then, Hen
             | exact (pending_ok_step _ _ _ _ _ Hp Hpend) ].
        -- assert (Hp' : is_pending_projection en' x d = false)
             by (unfold is_pending_projection in *; rewrite Hps; exact Hp).
           destruct (named_is_set d) eqn:Hs.
           ++ rewrite (rebuild_cc_let_set F sol cc cds b ucids codes ff None en
                         x kind d body Hp Hs).
              rewrite (rebuild_cc_let_set F' sol' cc cds b ucids codes ff None
                         en' x kind d (erase [] body) Hp' Hs).
              rewrite (projections_of_set_erase _ _ _ Hset).
              rewrite dead_eq.
              rewrite (existsb_ext_in (fun y => is_dead_var sol' y Value)
                         (fun y => is_dead_var sol y Value))
                by (intros; apply dead_eq).
              rewrite used_eq.
              rewrite (existsb_ext_in (fun y => is_var_used sol' y Value)
                         (fun y => is_var_used sol y Value))
                by (intros; apply used_eq).
              destruct (_ || _); [ reflexivity | ].
              destruct (_ || _).
              ** rewrite (erase_let_plain _ _ _ _ _
                            (rebuild_set_plain _ _ _ _ Hrp)).
                 rewrite (rebuild_set_eq sol sol' Hu Hfu _ _ Hen _ _ Hd).
                 unfold set_closure_nodes. rewrite HC. f_equal.
                 apply IH; [ exact Hla | apply env_rel_bind_then, Hen
                           | apply pending_ok_set ].
              ** apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
           ++ rewrite (rebuild_cc_let_other F sol cc cds b ucids codes ff None
                         en x kind d body Hp Hs).
              rewrite (rebuild_cc_let_other F' sol' cc cds b ucids codes ff
                         None en' x kind d (erase [] body) Hp' Hs).
              rewrite dead_eq, used_eq.
              destruct (is_dead_var sol x kind); [ reflexivity | ].
              destruct (is_var_used sol x kind).
              ** rewrite (erase_let_plain _ _ _ _ _
                            (rebuild_named_plain _ _ _ _ Hrp)).
                 rewrite (rebuild_named_eq sol sol' Hu Hfu _ _ Hen _ _ Hd).
                 f_equal.
                 apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
              ** apply IH; [ exact Hla | apply env_rel_bind, Hen
                           | apply pending_ok_bind ].
  - (* Let_static *)
    intros bindings body Hcodes IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    rewrite !rebuild_cc_let_static. cbn [erase].
    rewrite rebuild_cc_let_static. f_equal.
    + clear IH Hb Hpend.
      induction bindings as [ | bd bds IHb]; [ reflexivity | ].
      cbn [forallb] in Hbs. apply andb_true_iff in Hbs as [Hbd Hbs].
      cbn [flat_map map]. rewrite map_app. f_equal.
      * destruct bd as [id c | id | sym const].
        -- destruct c as [params mc ret exn md cbody].
           cbn [erase_binding erase_code].
           rewrite !rebuild_binding_cc_code.
           rewrite (code_id_used_eq sol sol' Hu).
           destruct (is_code_id_used sol id); cbn [map]; [ | reflexivity ].
           rewrite !rebuild_code_cc_unfold, code_change_eq.
           cbn [erase_binding]. rewrite erase_cc_code_of. do 3 f_equal.
           rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
                      (fun '(x, kind) => is_dead_var sol x kind))
             by (intros [x k] _; apply dead_eq).
           match goal with
           |- context [ if existsb ?f ?l then _ else _ ] =>
             destruct (existsb f l)
           end; [ reflexivity | ].
           cbn [binding_regions_plain code_regions_plain] in Hbd.
           apply andb_true_iff in Hbd as [_ Hcb].
           apply (Hcodes id (Code params mc ret exn md cbody)
                    (or_introl eq_refl));
             [ | apply env_rel_refl | apply pending_ok_bind ].
           cbn [code_body bind_variables bind_variables_then env_kinds rev].
           rewrite cc_code_env_kinds, app_nil_r. exact Hcb.
        -- reflexivity.
        -- cbn [rebuild_binding_cc erase_binding].
           rewrite (static_const_is_used_eq sol sol' Hu).
           destruct (static_const_is_used sol sym const); [ | reflexivity ].
           cbn [map erase_binding]. cbn [binding_regions_plain] in Hbd.
           rewrite (rebuild_static_const_eq sol sol' Hu Hfu _ _ Hen _ _ Hbd).
           reflexivity.
      * apply IHb; [ | exact Hbs ].
        intros id c Hin. apply (Hcodes id c). right. exact Hin.
    + apply IH; [ exact Hb | apply env_rel_bind, Hen | apply pending_ok_bind ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    rewrite !rebuild_cc_let_cont.
    rewrite (handler_exn_erase h [] Hhk), (handler_mask_erase _ _ h [] Hhk),
      (handler_mask_eq sol sol' Hu).
    set (mask := handler_mask sol (handler_is_exn_handler h) h).
    assert (IHb : erase [] (RB (bind_continuations en [(k, mask)]) body)
                  = RB' (bind_continuations en' [(k, mask)]) (erase [] body))
      by (apply IH; [ exact Hb | apply env_rel_bind_continuations, Hen
                    | apply pending_ok_nil; reflexivity ]).
    rewrite <- IHb, (proj1 free_continuations_erase).
    destruct (among k (free_continuations_of_expr
                         (RB (bind_continuations en [(k, mask)]) body)));
      [ | reflexivity ].
    cbn [erase]. f_equal.
    + apply IHh; assumption.
    + apply erase_plain, conts_plain_cons; [ | apply conts_plain_nil ].
      apply rebuilt_handler_places_cc, Hhk.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    rewrite rebuild_cc_let_cont_rec. cbn [erase].
    rewrite rebuild_cc_let_cont_rec.
    rewrite (masks_erase sol' _ handlers Hplain).
    rewrite (masks_eq sol sol' Hu handlers).
    set (en1 := bind_continuations en
                  (map (fun '(k, h) => (k, handler_mask sol false h))
                     handlers)).
    set (en1' := bind_continuations en'
                   (map (fun '(k, h) => (k, handler_mask sol false h))
                      handlers)).
    assert (Hen1 : env_rel en1 en1')
      by (apply env_rel_bind_continuations, Hen).
    f_equal.
    + rewrite !map_map. apply map_ext_in.
      intros [k h] Hin. cbn beta iota.
      rewrite (erase_handler_plain _ h)
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply region_places_plain; exact (Hplain k0 h0 Hin0)).
      rewrite (erase_handler_plain _ (rebuild_handler_cc _ _ _ _ _ _ _ _ _ _
                                      _ _ _))
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply rebuilt_handler_places_cc; exact (Hplain k0 h0 Hin0)).
      rewrite (handler_mask_erase _ _ h [] (Hplain k h Hin)),
        (handler_mask_eq sol sol' Hu).
      f_equal. apply IHh with (k := k); [ exact Hin | | exact Hen1 ].
      exact (Hhs (k, h) Hin).
    + rewrite (erase_plain _ body) by (apply rec_conts_plain, Hplain).
      rewrite (erase_plain _ (RB en1 body))
        by (apply rebuilt_conts_plain_cc, Hplain).
      apply IH; [ exact Hb | exact Hen1 | apply pending_ok_nil; reflexivity ].
  - (* Apply *)
    intros ap en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Hr].
    cbn [rebuild_expr_cc erase]. rewrite (erase_rebuild_apply_cc _ _ Hr).
    symmetry. exact (rebuild_apply_cc_eq en en' Hen ap Hs).
  - (* Apply_cont *)
    intros ac en en' Hpl Hen Hpend. cbn [regions_plain] in Hpl.
    cbn [rebuild_expr_cc erase]. rewrite erase_rebuild_apply_cont,
      erase_apply_cont_nil.
    symmetry. exact (rebuild_apply_cont_eq sol sol' Hu Hsrc en en' Hen ac Hpl).
  - (* Switch *)
    intros scrutinee arms en en' Hpl Hen Hpend.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Harms].
    cbn [rebuild_expr_cc erase]. rewrite erase_rebuild_switch, map_arms_id.
    symmetry.
    exact (rebuild_switch_eq sol sol' Hu Hsrc en en' Hen scrutinee arms Hs
             Harms).
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en en' dc mask Hpl Hen.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    rewrite !rebuild_handler_cc_unfold.
    rewrite erase_handler_no_region by (apply kept_metadata_plain, Hk).
    f_equal.
    rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
               (fun '(x, kind) => is_dead_var sol x kind))
      by (intros [x k] _; apply dead_eq).
    destruct (dc && _); [ reflexivity | ].
    apply IH; [ exact Hb | apply env_rel_bind, Hen | apply pending_ok_bind ].
Qed.

End CommuteCc.
