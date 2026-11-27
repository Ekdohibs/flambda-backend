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

(* What the rebuild keeps of a unit on the fragment
   FlambdaReaperRegionsSpec.regions_plain: no symbol that the unit has
   not, no poison of the kind [Region] that the unit has not, and no
   handler with a parameter of that kind; so the rebuilt unit is erasable
   (FlambdaRegionsSpec.erasable) and has no symbol at or above a base
   wherever the unit is and has not. Proved of the rebuild with changes
   of calling convention, for any table, where the code that changes
   returns no region; the plain rebuild is that rebuild where no code
   changes (FlambdaReaperRebuildProofs.rebuild_expr_cc_unchanged).

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsProofs.v and FlambdaReaperRegionsCcProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaValueSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaRegionsSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsRebuildProofs
  FlambdaReaperRegionsTraverseProofs.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaReaperFlowProofs
  FlambdaReaperRebuildProofs.
Import ListNotations.

(* ======================================================================== *)
(* The simples of a term                                                    *)
(* ======================================================================== *)

Section Simples.

(* What every simple, every symbol bound and the kinds of the parameters
   of every handler are to meet. *)
Variable P : simple -> bool.
Variable Q : symbol -> bool.
Variable H : list flambda_kind -> bool.

Fixpoint simples_ok (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    forallb P (simples_of_named defining_expr) && simples_ok body
  | Let_static bindings body =>
    forallb binding_simples_ok bindings && simples_ok body
  | Let_cont_non_recursive _ h body => handler_simples_ok h && simples_ok body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_simples_ok h) handlers
    && simples_ok body
  | Apply ap => forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap)
  | Apply_cont ac => forallb P (ac_args ac)
  | Switch scrutinee arms =>
    P scrutinee && forallb (fun '(_, ac) => forallb P (ac_args ac)) arms
  | Invalid => true
  end
with handler_simples_ok (h : continuation_handler) : bool :=
  match h with
  | Handler _ metadata handler =>
    H (hm_params_kinds metadata) && simples_ok handler
  end
with binding_simples_ok (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_simples_ok c
  | Bind_deleted_code _ => true
  | Bind_symbol s const =>
    Q s
    && (forallb P (FlambdaRegionsSpec.simples_of_static_const const)
        && match const with
           | Static_const_closure_of_set _ closure_symbols _ _ =>
             forallb (fun '(_, s) => Q s) closure_symbols
           | _ => true
           end)
  end
with code_simples_ok (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => simples_ok body
  end.

End Simples.

Lemma forallb_flat_map {A B : Type} (f : B -> bool) (g : A -> list B)
    (l : list A) :
  forallb f (flat_map g l) = forallb (fun a => forallb f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ].
  cbn. rewrite forallb_app, IH. reflexivity.
Qed.

Lemma forallb_true_pairs {A B : Type} (l : list (A * B)) :
  forallb (fun '(_, _) => true) l = true.
Proof. induction l as [ | [a b] l IH]; [ reflexivity | exact IH ]. Qed.

Lemma expr_below_simples (base : symbol) :
  (∀ e : expr,
     expr_below base e
     = simples_ok (simple_below base) (symbol_below base) (fun _ => true) e)
  ∧ (∀ h : continuation_handler,
       handler_below base h
       = handler_simples_ok (simple_below base) (symbol_below base)
           (fun _ => true) h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH. cbn. rewrite IH. reflexivity.
  - intros bindings body Hcodes IH. cbn [expr_below simples_ok].
    rewrite IH. f_equal. apply forallb_ext_in. intros [id c | id | s const] Hin;
      [ | reflexivity | reflexivity ].
    destruct c as [params mc ret exn md cbody].
    exact (Hcodes id _ Hin).
  - intros k h body IHh IH. cbn [expr_below simples_ok].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH. cbn [expr_below simples_ok].
    rewrite IH. f_equal. apply forallb_ext_in. intros [k h] Hin.
    exact (IHh k h Hin).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH. cbn. exact IH.
Qed.

Lemma no_region_poison_simples :
  (∀ e : expr,
     no_region_poison e
     = simples_ok simple_no_region_poison (fun _ => true) (fun _ => true) e)
  ∧ (∀ h : continuation_handler,
       handler_no_region_poison h
       = handler_simples_ok simple_no_region_poison (fun _ => true)
           (fun _ => true) h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH. cbn. rewrite IH. reflexivity.
  - intros bindings body Hcodes IH. cbn [no_region_poison simples_ok].
    rewrite IH. f_equal. apply forallb_ext_in. intros [id c | id | s const] Hin;
      [ | reflexivity | ].
    + destruct c as [params mc ret exn md cbody].
      exact (Hcodes id _ Hin).
    + cbn [binding_no_region_poison binding_simples_ok andb].
      destruct (forallb simple_no_region_poison _); [ | reflexivity ].
      destruct const; try reflexivity. cbn.
      rewrite forallb_true_pairs. reflexivity.
  - intros k h body IHh IH. cbn [no_region_poison simples_ok].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH. cbn [no_region_poison simples_ok].
    rewrite IH. f_equal. apply forallb_ext_in. intros [k h] Hin.
    exact (IHh k h Hin).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH. cbn. exact IH.
Qed.

(* ======================================================================== *)
(* Handlers with no parameter of the kind [Region]                          *)
(* ======================================================================== *)

Lemma regions_state_nothing_plain (places : list bool) :
  forallb negb places = true ->
  ∀ sks : list param_subkind, regions_state_nothing places sks = true.
Proof.
  induction places as [ | p places IH]; intros Hp sks;
    [ destruct sks; reflexivity | ].
  cbn in Hp. apply andb_true_iff in Hp as [Hp Hps].
  destruct sks as [ | sk sks]; [ reflexivity | ].
  cbn. destruct p; [ discriminate Hp | ]. cbn. exact (IH Hps sks).
Qed.

Lemma takes_no_region_plain (c : region_conts) (k : continuation) :
  conts_plain c -> takes_no_region c k = true.
Proof.
  intros Hc. unfold takes_no_region.
  destruct (lookup k c) as [p | ] eqn:Hl; [ | reflexivity ].
  pose proof (Hc k p Hl) as Hp. apply negb_true_iff.
  apply not_true_iff_false. intros E. apply existsb_exists in E as [q [Hq Eq]].
  subst q. rewrite forallb_forall in Hp. specialize (Hp true Hq).
  discriminate Hp.
Qed.

Definition no_region_kinds (kinds : list flambda_kind) : bool :=
  forallb kind_not_region kinds.

Lemma calls_return_simples :
  (∀ e : expr, ∀ c : region_conts,
     conts_plain c ->
     simples_ok (fun _ => true) (fun _ => true) no_region_kinds e = true ->
     calls_return_no_region c e = true)
  ∧ (∀ h : continuation_handler, ∀ c : region_conts,
       conts_plain c ->
       handler_simples_ok (fun _ => true) (fun _ => true) no_region_kinds h
       = true ->
       handler_returns_no_region c h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c Hc Hs. cbn [simples_ok] in Hs.
    apply andb_true_iff in Hs as [_ Hs]. exact (IH c Hc Hs).
  - intros bindings body Hcodes IH c Hc Hs.
    cbn [simples_ok] in Hs. apply andb_true_iff in Hs as [Hbs Hb].
    cbn [calls_return_no_region]. rewrite (IH c Hc Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros [id cd | id | s const] Hin;
      [ | reflexivity | reflexivity ].
    destruct cd as [params mc ret exn md cbody].
    specialize (Hbs _ Hin). cbn in Hbs |- *.
    exact (Hcodes id _ Hin [] conts_plain_nil Hbs).
  - intros k h body IHh IH c Hc Hs.
    cbn [simples_ok] in Hs. apply andb_true_iff in Hs as [Hh Hb].
    cbn [calls_return_no_region]. rewrite (IHh c Hc Hh). cbn [andb].
    apply IH; [ | exact Hb ].
    apply conts_plain_cons; [ | exact Hc ].
    destruct h as [params md hbody]. cbn in Hh |- *.
    apply andb_true_iff in Hh as [Hk _]. apply region_places_plain, Hk.
  - intros handlers body IHh IH c Hc Hs.
    cbn [simples_ok] in Hs. apply andb_true_iff in Hs as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hc' : conts_plain
                    (map (fun '(k, h) => (k, region_params h)) handlers
                     ++ c)).
    { apply conts_plain_app; [ | exact Hc ].
      intros k p Hin. apply in_map_iff in Hin as [[k' h] [Heq Hin]].
      injection Heq as <- <-. specialize (Hhs _ Hin).
      destruct h as [params md hbody]. cbn in Hhs |- *.
      apply andb_true_iff in Hhs as [Hk _]. apply region_places_plain, Hk. }
    cbn [calls_return_no_region]. rewrite (IH _ Hc' Hb), andb_true_r.
    apply forallb_forall. intros [k h] Hin.
    exact (IHh k h Hin _ Hc' (Hhs _ Hin)).
  - intros ap c Hc _. cbn.
    rewrite !takes_no_region_plain by exact Hc. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c Hc Hs. cbn in Hs |- *.
    apply andb_true_iff in Hs as [Hk Hb].
    rewrite regions_state_nothing_plain, (IH c Hc Hb); [ reflexivity | ].
    apply region_places_plain, Hk.
Qed.

(* ======================================================================== *)
(* The calls of a term of the fragment                                      *)
(* ======================================================================== *)

(* What the rebuild needs of a term to make no poison of the kind
   [Region] and no handler with a parameter of that kind: no argument of
   a call is a variable of the kind [Region], no result of a call is of
   that kind, nor is any parameter of a handler. [kinds] is as for
   [regions_plain]. *)
Fixpoint calls_plain (kinds : finite_map flambda_kind) (e : expr) : bool :=
  match e with
  | Let x kind _ body => calls_plain ((x, kind) :: kinds) body
  | Let_static bindings body =>
    forallb binding_calls_plain bindings && calls_plain kinds body
  | Let_cont_non_recursive _ h body =>
    handler_calls_plain kinds h && calls_plain kinds body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_calls_plain kinds h) handlers
    && calls_plain kinds body
  | Apply ap =>
    forallb (simple_not_region kinds) (ap_args ap)
    && forallb kind_not_region (unarize (ap_return_arity ap))
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_calls_plain (kinds : finite_map flambda_kind)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    forallb kind_not_region (hm_params_kinds metadata)
    && calls_plain
         (rev (with_kinds params (hm_params_kinds metadata)) ++ kinds) handler
  end
with binding_calls_plain (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_calls_plain c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_calls_plain (c : code) : bool :=
  match c with
  | Code params my_closure _ _ metadata body =>
    calls_plain
      (rev (with_kinds params (unarize (cm_params_arity metadata)))
       ++ [(my_closure, Value)])
      body
  end.

Lemma regions_plain_calls :
  (∀ e : expr, ∀ kinds,
     regions_plain kinds e = true -> calls_plain kinds e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds,
       handler_regions_plain kinds h = true ->
       handler_calls_plain kinds h = true).
Proof.
  apply regions_ind.
  - intros x kind d body IH IH2 kinds Hpl.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl;
        apply andb_true_iff in Hpl as [Hk Hb]; cbn [calls_plain];
        try (apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk; subst kind);
        exact (IH _ Hb).
    + rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [_ Hla]. unfold la_check in Hla.
      cbn [calls_plain].
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab;
        [ | exact (IH _ Hla) ].
      apply la_body_some in Hlab. subst body.
      apply andb_true_iff in Hla as [Hla Hb'].
      apply andb_true_iff in Hla as [Hla _].
      apply andb_true_iff in Hla as [Hla Hk'].
      apply andb_true_iff in Hla as [_ Hy].
      apply Pos.eqb_eq in Hy. subst y.
      apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
      cbn [calls_plain]. exact (IH2 _ _ _ _ eq_refl _ Hb').
  - intros bindings body Hcodes IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    cbn [calls_plain]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros [id c | id | s const] Hin;
      [ | reflexivity | reflexivity ].
    destruct c as [params mc ret exn md cbody].
    specialize (Hbs _ Hin). cbn in Hbs |- *.
    apply andb_true_iff in Hbs as [_ Hcb].
    exact (Hcodes id _ Hin _ Hcb).
  - intros k h body IHh IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    cbn [calls_plain]. rewrite (IHh _ Hh), (IH _ Hb). reflexivity.
  - intros handlers body IHh IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    cbn [calls_plain]. rewrite (IH _ Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap kinds Hpl. cbn [regions_plain] in Hpl.
    apply andb_true_iff in Hpl as [Hs Hr]. cbn [forallb] in Hs.
    apply andb_true_iff in Hs as [_ Hs]. rewrite forallb_app in Hs.
    apply andb_true_iff in Hs as [Hs _]. cbn. rewrite Hs, Hr. reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH kinds Hpl. cbn in Hpl |- *.
    apply andb_true_iff in Hpl as [Hk Hb]. rewrite Hk. exact (IH _ Hb).
Qed.

Lemma forallb_true {A : Type} (l : list A) : forallb (fun _ => true) l = true.
Proof. induction l as [ | a l IH]; [ reflexivity | exact IH ]. Qed.

(* A term of [calls_plain] has no handler with a parameter of the kind
   [Region]. *)
Lemma calls_plain_handlers :
  (∀ e : expr, ∀ kinds,
     calls_plain kinds e = true ->
     simples_ok (fun _ => true) (fun _ => true) no_region_kinds e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds,
       handler_calls_plain kinds h = true ->
       handler_simples_ok (fun _ => true) (fun _ => true) no_region_kinds h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds Hc. cbn in Hc |- *.
    rewrite forallb_true. exact (IH _ Hc).
  - intros bindings body Hcodes IH kinds Hc. cbn [calls_plain] in Hc.
    apply andb_true_iff in Hc as [Hbs Hb]. cbn [simples_ok].
    rewrite (IH _ Hb), andb_true_r. rewrite forallb_forall in Hbs |- *.
    intros [id c | id | s const] Hin.
    + destruct c as [params mc ret exn md cbody]. cbn.
      exact (Hcodes id _ Hin _ (Hbs _ Hin)).
    + reflexivity.
    + cbn. rewrite forallb_true. destruct const; try reflexivity.
      apply forallb_true_pairs.
  - intros k h body IHh IH kinds Hc. cbn [calls_plain] in Hc.
    apply andb_true_iff in Hc as [Hh Hb]. cbn [simples_ok].
    apply andb_true_iff. split; [ exact (IHh _ Hh) | exact (IH _ Hb) ].
  - intros handlers body IHh IH kinds Hc. cbn [calls_plain] in Hc.
    apply andb_true_iff in Hc as [Hhs Hb]. cbn [simples_ok].
    rewrite (IH _ Hb), andb_true_r. rewrite forallb_forall in Hhs |- *.
    intros [k h] Hin. exact (IHh k h Hin _ (Hhs _ Hin)).
  - intros ap kinds _. apply forallb_true.
  - intros ac kinds _. apply forallb_true.
  - intros scrutinee arms kinds _. cbn [simples_ok].
    apply forallb_forall. intros [i ac] _. apply forallb_true.
  - reflexivity.
  - intros params md hbody IH kinds Hc. cbn in Hc |- *.
    apply andb_true_iff in Hc as [Hk Hb]. unfold no_region_kinds.
    rewrite Hk. exact (IH _ Hb).
Qed.

(* ======================================================================== *)
(* The rebuild with changes of calling convention, case by case             *)
(* ======================================================================== *)

(* The environment of the body of code that changes its calling
   convention or not, and the code rebuilt. *)
Definition cc_code_env (change : calling_convention_change)
    (ret : continuation) : rebuild_env :=
  match change with
  | Not_changing_calling_convention => FlambdaReaperRebuild.empty_env
  | Changing_calling_convention _ return_decisions =>
    bind_continuations FlambdaReaperRebuild.empty_env
      [(ret, map is_kept return_decisions)]
  end.

Definition cc_code_of (types : subkinds_rewriting) (id : code_id)
    (change : calling_convention_change) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) : code :=
  match change with
  | Not_changing_calling_convention =>
    Code params mc ret exn (kept_code_metadata types id md) body
  | Changing_calling_convention params_decisions return_decisions =>
    Code (filter_kept (map is_kept params_decisions) params) mc ret exn
      (changed_code_metadata_with types id md params_decisions
         return_decisions)
      body
  end.

Lemma cc_code_env_kinds (change : calling_convention_change)
    (ret : continuation) :
  env_kinds (cc_code_env change ret) = [].
Proof. destruct change; reflexivity. Qed.

Lemma cc_code_of_body (types : subkinds_rewriting) (id : code_id)
    (change : calling_convention_change) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (body : expr) :
  code_body (cc_code_of types id change params mc ret exn md body) = body.
Proof. destruct change; reflexivity. Qed.

Section UnfoldCc.

Variables (F0 : facts) (sol0 : solution) (cc0 : node -> bool)
  (cds0 : finite_map code_dep) (b0 : bool) (ucids0 : list code_id)
  (codes0 : finite_map code_metadata) (ff0 : positive)
  (types0 : subkinds_rewriting).

Abbreviation RBc := (rebuild_expr_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0
                       types0).
Abbreviation RHc := (rebuild_handler_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0
                       types0).

Lemma rebuild_cc_let_pending (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = true ->
  RBc en (Let x kind d body)
  = Let x kind d
      (RBc (bind_variables_then en [(x, kind)] (tl (env_projections en)))
         body).
Proof. intros H. cbn [rebuild_expr_cc]. rewrite H. reflexivity. Qed.

Lemma rebuild_cc_let_set (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = false -> named_is_set d = true ->
  RBc en (Let x kind d body)
  = if is_dead_var sol0 x kind
       || existsb (fun y => is_dead_var sol0 y Value)
            (projections_of_set x d body)
    then Invalid
    else if is_var_used sol0 x kind
            || existsb (fun y => is_var_used sol0 y Value)
                 (projections_of_set x d body)
    then
      Let x kind
        (rebuild_set sol0 en (set_closure_nodes F0 (Node_variable x)) d)
        (RBc (bind_variables_then en [(x, kind)]
                (projections_of_set x d body))
           body)
    else RBc (bind_variables en [(x, kind)]) body.
Proof.
  intros H Hs. cbn [rebuild_expr_cc]. rewrite H.
  destruct d; try discriminate Hs; reflexivity.
Qed.

Lemma rebuild_cc_let_other (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  is_pending_projection en x d = false -> named_is_set d = false ->
  RBc en (Let x kind d body)
  = if is_dead_var sol0 x kind then Invalid
    else if is_var_used sol0 x kind
    then
      Let x kind (rebuild_named sol0 en x d)
        (RBc (bind_variables en [(x, kind)]) body)
    else RBc (bind_variables en [(x, kind)]) body.
Proof.
  intros H Hs. cbn [rebuild_expr_cc]. rewrite H.
  destruct d; try discriminate Hs; reflexivity.
Qed.

Lemma rebuild_cc_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  RBc en (Let_cont_recursive handlers body)
  = Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, RHc (bind_continuations en
                       (map (fun '(k, h) => (k, handler_mask sol0 false h))
                          handlers))
                  false (handler_mask sol0 false h) h))
         handlers)
      (RBc (bind_continuations en
              (map (fun '(k, h) => (k, handler_mask sol0 false h)) handlers))
         body).
Proof. reflexivity. Qed.

Lemma rebuild_cc_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  RBc en (Let_cont_non_recursive k h body)
  = if among k
         (free_continuations_of_expr
            (RBc (bind_continuations en
                    [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
               body))
    then
      Let_cont_non_recursive k
        (RHc en true (handler_mask sol0 (handler_is_exn_handler h) h) h)
        (RBc (bind_continuations en
                [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
           body)
    else
      RBc (bind_continuations en
             [(k, handler_mask sol0 (handler_is_exn_handler h) h)])
        body.
Proof. reflexivity. Qed.

Lemma rebuild_cc_let_static (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  RBc en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0 types0 en)
         bindings)
      (RBc (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma rebuild_handler_cc_unfold (en : rebuild_env) (dc : bool)
    (mask : list bool) (params : list variable) (md : handler_metadata)
    (hbody : expr) :
  RHc en dc mask (Handler params md hbody)
  = Handler (filter_kept mask params)
      (kept_handler_metadata types0 mask params md)
      (if dc
          && existsb (fun '(x, kind) => is_dead_var sol0 x kind)
               (with_kinds params (hm_params_kinds md))
       then Invalid
       else RBc (bind_variables en (with_kinds params (hm_params_kinds md)))
              hbody).
Proof. reflexivity. Qed.

Lemma rebuild_binding_cc_code (en : rebuild_env) (id : code_id) (c : code) :
  rebuild_binding_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0 types0 en
    (Bind_code id c)
  = if is_code_id_used sol0 id
    then
      [Bind_code id
         (rebuild_code_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0 types0 id c)]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma rebuild_code_cc_unfold (id : code_id) (params : list variable)
    (mc : variable) (ret exn : continuation) (md : code_metadata)
    (cbody : expr) :
  rebuild_code_cc F0 sol0 cc0 cds0 b0 ucids0 codes0 ff0 types0 id
    (Code params mc ret exn md cbody)
  = cc_code_of types0 id
      (code_calling_convention_change sol0 cc0 cds0 ucids0 id md)
      params mc ret exn md
      (if existsb (fun '(x, kind) => is_dead_var sol0 x kind)
            ((mc, Value) :: with_kinds params (unarize (cm_params_arity md)))
       then Invalid
       else
         RBc (bind_variables
                (cc_code_env
                   (code_calling_convention_change sol0 cc0 cds0 ucids0 id
                      md)
                   ret)
                ((mc, Value)
                 :: with_kinds params (unarize (cm_params_arity md))))
           cbody).
Proof.
  cbn [rebuild_code_cc]. unfold cc_code_of, cc_code_env.
  destruct (code_calling_convention_change sol0 cc0 cds0 ucids0 id md);
    reflexivity.
Qed.

End UnfoldCc.

(* ======================================================================== *)
(* The rebuild keeps the simples                                            *)
(* ======================================================================== *)

Section SimplesEq.

Variables (P : simple -> bool) (Q : symbol -> bool)
  (H : list flambda_kind -> bool).

Lemma so_let (x : variable) (kind : flambda_kind) (d : named) (body : expr) :
  simples_ok P Q H (Let x kind d body)
  = forallb P (simples_of_named d) && simples_ok P Q H body.
Proof. reflexivity. Qed.

Lemma so_let_static (bindings : list static_binding) (body : expr) :
  simples_ok P Q H (Let_static bindings body)
  = forallb (binding_simples_ok P Q H) bindings && simples_ok P Q H body.
Proof. reflexivity. Qed.

Lemma so_let_cont (k : continuation) (h : continuation_handler)
    (body : expr) :
  simples_ok P Q H (Let_cont_non_recursive k h body)
  = handler_simples_ok P Q H h && simples_ok P Q H body.
Proof. reflexivity. Qed.

Lemma so_let_cont_rec (handlers : list (continuation * continuation_handler))
    (body : expr) :
  simples_ok P Q H (Let_cont_recursive handlers body)
  = forallb (fun '(_, h) => handler_simples_ok P Q H h) handlers
    && simples_ok P Q H body.
Proof. reflexivity. Qed.

Lemma so_handler (params : list variable) (md : handler_metadata)
    (hbody : expr) :
  handler_simples_ok P Q H (Handler params md hbody)
  = H (hm_params_kinds md) && simples_ok P Q H hbody.
Proof. reflexivity. Qed.

Lemma so_binding_code (id : code_id) (c : code) :
  binding_simples_ok P Q H (Bind_code id c) = simples_ok P Q H (code_body c).
Proof. destruct c; reflexivity. Qed.

End SimplesEq.

Lemma forallb_map_fst {A B : Type} (f : A -> bool) (l : list (A * B)) :
  forallb f (map fst l) = forallb (fun p => f (fst p)) l.
Proof.
  induction l as [ | [a b] l IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma kept_handler_metadata_kinds (mask : list bool) (params : list variable)
    (md : handler_metadata) :
  hm_params_kinds (kept_handler_metadata None mask params md)
  = filter_kept mask (hm_params_kinds md).
Proof.
  cbn [kept_handler_metadata]. unfold with_params_kinds.
  destruct (hm_is_exn_handler md); reflexivity.
Qed.

Lemma kept_kinds_in (decisions : list param_decision) (k : flambda_kind) :
  k ∈ kept_kinds decisions -> Keep k ∈ decisions.
Proof.
  unfold kept_kinds. intros Hin. apply in_flat_map in Hin as [d [Hd Hk]].
  destruct d as [k' | ]; [ | destruct Hk ].
  destruct Hk as [<- | []]. exact Hd.
Qed.

Lemma kept_kinds_of_in (decisions : list param_decision) (k : flambda_kind) :
  Keep k ∈ decisions -> k ∈ kept_kinds decisions.
Proof.
  unfold kept_kinds. intros Hin. apply in_flat_map.
  exists (Keep k). split; [ exact Hin | left; reflexivity ].
Qed.

(* The wrapper of a call of code that changes its calling convention
   gives the continuation the parameters of the wrapper and poison of the
   kinds of the results that the callee no longer returns. *)
Lemma wrapper_args_in (next : positive) (mask : list bool)
    (fd : list (param_decision * flambda_kind)) (s : simple) :
  s ∈ wrapper_args next mask fd ->
  (∃ v, s = Var v) ∨ (∃ kind, s = Const (Poison kind) ∧ (Delete, kind) ∈ fd).
Proof.
  revert next fd. induction mask as [ | keep mask IH]; intros next fd Hin;
    [ destruct fd; destruct Hin | ].
  destruct fd as [ | [d kind] fd]; [ destruct Hin | ].
  destruct d as [k | ]; cbn [wrapper_args] in Hin;
    apply in_app_or in Hin as [Hin | Hin].
  - destruct keep; [ | destruct Hin ]. destruct Hin as [<- | []].
    left. eexists. reflexivity.
  - destruct (IH _ _ Hin) as [Hv | (k' & Hs & Hk)]; [ left; exact Hv | ].
    right. exists k'. split; [ exact Hs | right; exact Hk ].
  - destruct keep; [ | destruct Hin ]. destruct Hin as [<- | []].
    right. exists kind. split; [ reflexivity | left; reflexivity ].
  - destruct (IH _ _ Hin) as [Hv | (k' & Hs & Hk)]; [ left; exact Hv | ].
    right. exists k'. split; [ exact Hs | right; exact Hk ].
Qed.

Section Keeps.

Variable P : simple -> bool.
Variable Q : symbol -> bool.
Variable H : list flambda_kind -> bool.

Hypothesis HP_var : ∀ x : variable, P (Var x) = true.
Hypothesis HP_const :
  ∀ c : const, const_no_region_poison c = true -> P (Const c) = true.
Hypothesis HH_plain :
  ∀ l : list flambda_kind, forallb kind_not_region l = true -> H l = true.
Hypothesis HH_filter :
  ∀ (mask : list bool) (l : list flambda_kind),
    H l = true -> H (filter_kept mask l) = true.

Variables (F : facts) (sol : solution) (cannot_change : node -> bool)
  (code_deps : finite_map code_dep) (b : bool) (ucids : list code_id)
  (codes : finite_map code_metadata) (ff : positive).

(* The code that a call of the unit is known to call, if it changes its
   calling convention, returns no region. *)
Hypothesis Hres :
  ∀ (ap : apply_expr) (pd rd : list param_decision),
    apply_calling_convention_change F sol cannot_change code_deps ucids codes
      ap
    = Changing_calling_convention pd rd ->
    forallb kind_not_region (kept_kinds rd) = true.

Abbreviation SO := (simples_ok P Q H).
Abbreviation RB :=
  (rebuild_expr_cc F sol cannot_change code_deps b ucids codes ff None).
Abbreviation RH :=
  (rebuild_handler_cc F sol cannot_change code_deps b ucids codes ff None).

Lemma poison_unused (x : variable) (kind : flambda_kind) :
  is_var_used sol x kind = false ->
  const_no_region_poison (Poison kind) = true.
Proof. destruct kind; cbn; try reflexivity. intros E. discriminate E. Qed.

Lemma P_rewrite (en : rebuild_env) (s : simple) :
  P s = true -> P (rewrite_simple sol en s) = true.
Proof.
  intros Hs. destruct s as [x | sym | c]; cbn [rewrite_simple].
  - destruct (is_var_used sol x (kind_of_variable en x)) eqn:E;
      [ apply HP_var | apply HP_const; exact (poison_unused _ _ E) ].
  - destruct (is_symbol_used sol sym); [ exact Hs | ].
    apply HP_const. reflexivity.
  - exact Hs.
Qed.

Lemma P_rewrite_or_variable (en : rebuild_env) (s : simple) :
  P s = true -> P (rewrite_or_variable sol en s) = true.
Proof.
  intros Hs. destruct s as [x | sym | c]; cbn [rewrite_or_variable];
    [ | exact Hs | exact Hs ].
  destruct (is_var_used sol x (kind_of_variable en x)) eqn:E;
    [ apply HP_var | apply HP_const ].
  destruct (kind_of_variable en x) as [ | nk | | ]; cbn; try reflexivity.
  - destruct nk; reflexivity.
  - discriminate E.
Qed.

Lemma P_poison_argument (en : rebuild_env) (s : simple) :
  simple_not_region (env_kinds en) s = true -> P s = true ->
  P (poison_argument en s) = true.
Proof.
  intros Hr Hs. destruct s as [x | sym | c]; cbn [poison_argument];
    [ | apply HP_const; reflexivity | exact Hs ].
  apply HP_const. cbn in Hr |- *. unfold region_variable in Hr.
  unfold kind_of_variable. destruct (lookup x (env_kinds en)) as [k | ];
    [ exact Hr | reflexivity ].
Qed.

Lemma forallb_rewrite (en : rebuild_env) (l : list simple) :
  forallb P l = true -> forallb P (map (rewrite_simple sol en) l) = true.
Proof.
  intros Hl. apply forallb_forall. intros s' Hin.
  apply in_map_iff in Hin as [s [<- Hs]]. apply P_rewrite.
  rewrite forallb_forall in Hl. exact (Hl s Hs).
Qed.

Lemma forallb_rebuild_fields (en : rebuild_env) (base : node)
    (fields : list simple) :
  ∀ i : nat, forallb P fields = true ->
    forallb P (rebuild_fields sol en base i fields) = true.
Proof.
  induction fields as [ | s fields IH]; intros i Hf; [ reflexivity | ].
  cbn in Hf. apply andb_true_iff in Hf as [Hs Hf].
  cbn [rebuild_fields forallb]. rewrite (IH _ Hf), andb_true_r.
  destruct (field_used sol base (Field_block i Value));
    [ exact (P_rewrite en s Hs) | apply HP_const; reflexivity ].
Qed.

Lemma forallb_value_slots (en : rebuild_env) (bound : list node)
    (vs : list (value_slot * simple)) :
  forallb P (map snd vs) = true ->
  forallb P (map snd (rebuild_value_slots sol en bound vs)) = true.
Proof.
  unfold rebuild_value_slots. induction vs as [ | [w s] vs IH]; intros Hv;
    [ reflexivity | ].
  cbn in Hv. apply andb_true_iff in Hv as [Hs Hv].
  cbn [flat_map]. rewrite map_app, forallb_app, (IH Hv), andb_true_r.
  destruct (existsb _ bound); [ | reflexivity ].
  cbn. rewrite (P_rewrite en s Hs). reflexivity.
Qed.

Lemma named_ok (en : rebuild_env) (x : variable) (d : named) :
  forallb P (simples_of_named d) = true ->
  forallb P (simples_of_named (rebuild_named sol en x d)) = true.
Proof.
  intros Hd. destruct d as [s | p args | | | | | ];
    cbn [rebuild_named simples_of_named] in *.
  - cbn in Hd |- *. apply andb_true_iff in Hd as [Hs _].
    rewrite (P_rewrite en s Hs). reflexivity.
  - destruct p as [u | bp | vp | hp | rp];
      try (apply forallb_rewrite; exact Hd).
    + destruct u; try (apply forallb_rewrite; exact Hd).
      destruct (field_used sol _ _); [ apply forallb_rewrite; exact Hd | ].
      cbn. rewrite HP_const; reflexivity.
    + destruct vp; try (apply forallb_rewrite; exact Hd).
      apply forallb_rebuild_fields, Hd.
    + destruct hp; try (apply forallb_rewrite; exact Hd).
      apply forallb_rebuild_fields, Hd.
  - exact Hd.
  - exact Hd.
  - cbn in Hd |- *. apply andb_true_iff in Hd as [Hs _].
    rewrite (P_rewrite en _ Hs). reflexivity.
  - cbn in Hd |- *. apply andb_true_iff in Hd as [Hs _].
    rewrite (P_rewrite en _ Hs). reflexivity.
  - cbn in Hd |- *. apply andb_true_iff in Hd as [Hs _].
    rewrite (P_rewrite en _ Hs). reflexivity.
Qed.

Lemma set_ok (en : rebuild_env) (bound : list node) (d : named) :
  forallb P (simples_of_named d) = true ->
  forallb P (simples_of_named (rebuild_set sol en bound d)) = true.
Proof.
  intros Hd. destruct d; cbn [rebuild_set simples_of_named] in *;
    try exact Hd; apply forallb_value_slots, Hd.
Qed.

Lemma static_const_ok (en : rebuild_env) (s : symbol)
    (const : static_const) :
  binding_simples_ok P Q H (Bind_symbol s const) = true ->
  binding_simples_ok P Q H (Bind_symbol s (rebuild_static_const sol en s const))
  = true.
Proof.
  cbn [binding_simples_ok]. intros Hb.
  apply andb_true_iff in Hb as [Hq Hb]. apply andb_true_iff in Hb as [Hp Hc].
  rewrite Hq. cbn [andb].
  destruct const;
    cbn [rebuild_static_const FlambdaRegionsSpec.simples_of_static_const]
    in *.
  - rewrite (forallb_value_slots _ _ _ Hp). reflexivity.
  - rewrite (forallb_value_slots _ _ _ Hp). exact Hc.
  - rewrite (forallb_rebuild_fields _ _ _ _ Hp). reflexivity.
  - cbn in Hp |- *. apply andb_true_iff in Hp as [Hp _].
    rewrite (P_rewrite_or_variable en _ Hp). reflexivity.
  - reflexivity.
  - rewrite (forallb_rewrite _ _ Hp). reflexivity.
Qed.

Lemma rewrite_apply_cont_ok (en : rebuild_env) (ac ac' : apply_cont) :
  rewrite_apply_cont sol en ac = Some ac' ->
  forallb P (ac_args ac) = true -> forallb P (ac_args ac') = true.
Proof.
  unfold rewrite_apply_cont. destruct (existsb _ _); [ discriminate | ].
  intros E Hac. injection E as <-. cbn [ac_args].
  destruct (lookup (ac_continuation ac) (env_conts en));
    [ apply filter_kept_forallb | ]; apply forallb_rewrite, Hac.
Qed.

Lemma apply_cont_ok (en : rebuild_env) (ac : apply_cont) :
  forallb P (ac_args ac) = true -> SO (rebuild_apply_cont sol en ac) = true.
Proof.
  intros Hac. unfold rebuild_apply_cont.
  destruct (rewrite_apply_cont sol en ac) as [ac' | ] eqn:E;
    [ | reflexivity ].
  exact (rewrite_apply_cont_ok en ac ac' E Hac).
Qed.

Lemma switch_ok (en : rebuild_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  P scrutinee = true ->
  forallb (fun '(_, ac) => forallb P (ac_args ac)) arms = true ->
  SO (rebuild_switch sol en scrutinee arms) = true.
Proof.
  intros Hs Harms. unfold rebuild_switch.
  match goal with
  |- context [ Keyed.is_empty ?l ] => destruct (Keyed.is_empty l)
  end; [ reflexivity | ].
  cbn [simples_ok]. rewrite (P_rewrite en _ Hs). cbn [andb].
  rewrite forallb_flat_map. apply forallb_forall. intros [d ac] Hin.
  apply in_compact_arms in Hin. rewrite forallb_forall in Harms.
  specialize (Harms _ Hin). cbn beta iota in Harms.
  destruct (rewrite_apply_cont sol en ac) as [ac' | ] eqn:E;
    [ | reflexivity ].
  cbn. rewrite (rewrite_apply_cont_ok en ac ac' E Harms). reflexivity.
Qed.

Lemma exn_extra_ok (en : rebuild_env) (ap : apply_expr) :
  forallb P (exn_extra_simples ap) = true ->
  forallb P (map fst (rebuild_exn_extra_args sol en ap)) = true.
Proof.
  intros He. unfold rebuild_exn_extra_args.
  assert (Hx : forallb (fun p => P (fst p))
                 (map (fun '(s, kind) => (rewrite_simple sol en s, kind))
                    (ap_exn_extra_args ap)) = true).
  { unfold exn_extra_simples in He. rewrite forallb_map_fst in He.
    rewrite forallb_forall in He |- *. intros [s' k] Hin.
    apply in_map_iff in Hin as [[s k0] [Heq Hin]]. injection Heq as <- <-.
    exact (P_rewrite en s (He _ Hin)). }
  rewrite forallb_map_fst.
  destruct (lookup (ap_exn_continuation ap) (env_conts en));
    [ apply filter_kept_forallb | ]; exact Hx.
Qed.

Lemma mask_arguments_ok (en : rebuild_env) (args : list simple) :
  ∀ mask : list bool,
    forallb (simple_not_region (env_kinds en)) args = true ->
    forallb P args = true -> forallb P (mask_arguments en mask args) = true.
Proof.
  induction args as [ | s args IH]; intros mask Hr Ha; [ reflexivity | ].
  cbn in Hr, Ha. apply andb_true_iff in Hr as [Hrs Hr].
  apply andb_true_iff in Ha as [Hs Ha].
  destruct mask as [ | keep mask]; cbn [mask_arguments forallb];
    rewrite (IH _ Hr Ha), andb_true_r; [ exact Hs | ].
  destruct keep; [ exact Hs | exact (P_poison_argument en s Hrs Hs) ].
Qed.

Lemma arguments_ok (en : rebuild_env) (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en)) (ap_args ap) = true ->
  forallb P (ap_args ap) = true ->
  forallb P (rebuild_arguments F sol en ap) = true.
Proof.
  intros Hr Ha. unfold rebuild_arguments.
  destruct (ap_callee ap);
    [ apply mask_arguments_ok | apply mask_arguments_ok
    | apply forallb_rewrite ]; assumption.
Qed.

Lemma ap_simples (ap : apply_expr) :
  forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  P (ap_callee ap) = true ∧ forallb P (ap_args ap) = true
  ∧ forallb P (exn_extra_simples ap) = true.
Proof.
  intros Hs. cbn [forallb] in Hs. apply andb_true_iff in Hs as [Hc Hs].
  rewrite forallb_app in Hs. apply andb_true_iff in Hs as [Ha He].
  split; [ exact Hc | split; assumption ].
Qed.

Lemma wrapper_ok (en : rebuild_env) (ap : apply_expr) :
  forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  SO (make_apply_wrapper ff None en ap) = true.
Proof.
  intros Hs Hk. unfold make_apply_wrapper.
  destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ];
    [ | exact Hs ].
  destruct (forallb (fun keep => keep) mask); [ exact Hs | ].
  rewrite so_let_cont, so_handler. cbn [wrapper_metadata hm_params_kinds
    make_handler_metadata].
  rewrite (HH_plain _ Hk). cbn [andb simples_ok ac_args].
  rewrite filter_kept_forallb; [ exact Hs | ].
  apply forallb_forall. intros s Hin. apply in_map_iff in Hin as [v [<- _]].
  apply HP_var.
Qed.

Lemma rebuild_apply_ok (en : rebuild_env) (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en)) (ap_args ap) = true ->
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  SO (rebuild_apply F sol b ucids ff None en ap) = true.
Proof.
  intros Hr Hk Hs. destruct (ap_simples ap Hs) as (Hc & Ha & He).
  unfold rebuild_apply.
  destruct (_ || _); [ reflexivity | ]. apply wrapper_ok; [ | exact Hk ].
  cbn [forallb ap_callee ap_args exn_extra_simples ap_exn_extra_args].
  rewrite Hc, forallb_app, (arguments_ok en ap Hr Ha).
  exact (exn_extra_ok en ap He).
Qed.

Lemma wrapper_cc_ok (en : rebuild_env) (ap : apply_expr)
    (fd : list (param_decision * flambda_kind)) :
  forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  forallb kind_not_region (kept_kinds (map fst fd)) = true ->
  (∀ kind, (Delete, kind) ∈ fd -> kind_not_region kind = true) ->
  SO (make_apply_wrapper_cc ff en ap fd) = true.
Proof.
  intros Hs Hk Hd. unfold make_apply_wrapper_cc.
  match goal with
  | |- simples_ok _ _ _ (if ?c then _ else _) = true => destruct c
  end; [ exact Hs | ].
  rewrite so_let_cont, so_handler. cbn [hm_params_kinds make_handler_metadata].
  rewrite (HH_plain _ Hk). cbn [andb simples_ok ac_args].
  apply andb_true_iff. split; [ | exact Hs ].
  apply forallb_forall. intros s Hin.
  destruct (wrapper_args_in _ _ _ _ Hin) as [[v ->] | (kind & -> & Hin')];
    [ apply HP_var | ].
  apply HP_const. exact (Hd kind Hin').
Qed.

Lemma get_args_in (en : rebuild_env) (decisions : list param_decision)
    (args : list simple) (s : simple) :
  s ∈ map fst (get_args_with_kinds sol en decisions args) ->
  ∃ a, a ∈ args ∧ s = rewrite_simple sol en a.
Proof.
  unfold get_args_with_kinds. intros Hin.
  apply in_map_iff in Hin as [[s' k] [Heq Hin]]. cbn [fst] in Heq. subst s'.
  apply in_flat_map in Hin as [[a d] [Had Hin]].
  destruct d as [k' | ]; [ | destruct Hin ].
  destruct Hin as [Heq | []]. injection Heq as Heq _. subst s.
  exists a. split; [ exact (in_combine_l _ _ _ _ Had) | reflexivity ].
Qed.

Lemma cc_args_in (en : rebuild_env) (a : arity) (args : list simple)
    (pd : list param_decision) (s : simple) :
  s ∈ flat_map (map fst)
        (map (fun '(args, decisions) =>
                get_args_with_kinds sol en decisions args)
           (combine (group_by_parameter a args) (group_by_parameter a pd))) ->
  ∃ x, x ∈ args ∧ s = rewrite_simple sol en x.
Proof.
  intros Hin. apply in_flat_map in Hin as [g [Hg Hs]].
  apply in_map_iff in Hg as [[args' ds] [<- Hin]].
  destruct (get_args_in _ _ _ _ Hs) as (x & Hx & ->).
  exists x. split; [ | reflexivity ].
  apply in_combine_l in Hin. unfold group_by_parameter in Hin.
  exact (group_by_sizes_in _ _ _ _ Hin Hx).
Qed.

Lemma kept_kinds_combine (rd : list param_decision) (l : list flambda_kind) :
  forallb kind_not_region (kept_kinds rd) = true ->
  forallb kind_not_region (kept_kinds (map fst (combine rd l))) = true.
Proof.
  intros H0. rewrite forallb_forall in H0 |- *. intros k Hk.
  apply H0, kept_kinds_of_in. apply kept_kinds_in in Hk.
  apply in_map_iff in Hk as [[d k'] [Heq Hin]]. cbn [fst] in Heq. subst d.
  exact (in_combine_l _ _ _ _ Hin).
Qed.

Lemma rebuild_apply_cc_ok (en : rebuild_env) (ap : apply_expr) :
  forallb (simple_not_region (env_kinds en)) (ap_args ap) = true ->
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  forallb P (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  SO (rebuild_apply_cc F sol cannot_change code_deps b ucids codes ff None en
        ap)
  = true.
Proof.
  intros Hr Hk Hs. destruct (ap_simples ap Hs) as (Hc & Ha & He).
  unfold rebuild_apply_cc.
  destruct (apply_calling_convention_change F sol cannot_change code_deps
              ucids codes ap) as [ | pd rd] eqn:Hch;
    [ exact (rebuild_apply_ok en ap Hr Hk Hs) | ].
  destruct (_ || _); [ reflexivity | ].
  apply wrapper_cc_ok.
  - cbn [forallb ap_callee ap_args exn_extra_simples ap_exn_extra_args].
    rewrite Hc, forallb_app. cbn [andb]. apply andb_true_iff.
    split; [ | exact (exn_extra_ok en ap He) ].
    apply forallb_forall. intros s Hin.
    destruct (cc_args_in _ _ _ _ _ Hin) as (x & Hx & ->).
    apply P_rewrite. rewrite forallb_forall in Ha. exact (Ha x Hx).
  - apply kept_kinds_combine. exact (Hres ap pd rd Hch).
  - intros kind Hin. apply in_combine_r in Hin.
    rewrite forallb_forall in Hk. exact (Hk kind Hin).
Qed.

Lemma rebuild_cc_simples :
  (∀ e : expr, ∀ en : rebuild_env,
     calls_plain (env_kinds en) e = true -> SO e = true ->
     SO (RB en e) = true)
  ∧ (∀ h : continuation_handler,
       ∀ (en : rebuild_env) (dc : bool) (mask : list bool),
         handler_calls_plain (env_kinds en) h = true ->
         handler_simples_ok P Q H h = true ->
         handler_simples_ok P Q H (RH en dc mask h) = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en Hc Hs.
    cbn [calls_plain] in Hc. rewrite so_let in Hs.
    apply andb_true_iff in Hs as [Hd Hb].
    assert (IH' : ∀ en', env_kinds en' = (x, kind) :: env_kinds en ->
                    SO (RB en' body) = true)
      by (intros en' Hk; apply IH; [ rewrite Hk; exact Hc | exact Hb ]).
    destruct (is_pending_projection en x d) eqn:Hp.
    + rewrite (rebuild_cc_let_pending _ _ _ _ _ _ _ _ _ en x kind d body Hp).
      rewrite so_let, Hd, IH'; reflexivity.
    + destruct (named_is_set d) eqn:Hset.
      * rewrite (rebuild_cc_let_set _ _ _ _ _ _ _ _ _ en x kind d body Hp
                   Hset).
        destruct (_ || _); [ reflexivity | ].
        destruct (_ || _); [ | apply IH'; reflexivity ].
        rewrite so_let, set_ok, IH'; [ reflexivity | reflexivity | exact Hd ].
      * rewrite (rebuild_cc_let_other _ _ _ _ _ _ _ _ _ en x kind d body Hp
                   Hset).
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind); [ | apply IH'; reflexivity ].
        rewrite so_let, named_ok, IH'; [ reflexivity | reflexivity
                                        | exact Hd ].
  - (* Let_static *)
    intros bindings body Hcodes IH en Hc Hs.
    cbn [calls_plain] in Hc. apply andb_true_iff in Hc as [Hbc Hc].
    rewrite so_let_static in Hs. apply andb_true_iff in Hs as [Hbs Hb].
    rewrite rebuild_cc_let_static, so_let_static,
      (IH (bind_variables en []) Hc Hb), andb_true_r.
    rewrite forallb_flat_map. rewrite forallb_forall in Hbc, Hbs |- *.
    intros bd Hin. specialize (Hbc _ Hin). specialize (Hbs _ Hin).
    destruct bd as [id c | id | s const].
    + rewrite rebuild_binding_cc_code.
      destruct (is_code_id_used sol id); [ | reflexivity ].
      cbn [forallb]. rewrite andb_true_r.
      destruct c as [params mc ret exn md cbody].
      rewrite rebuild_code_cc_unfold, so_binding_code, cc_code_of_body.
      destruct (existsb _ _); [ reflexivity | ].
      apply (Hcodes id _ Hin).
      * cbn [bind_variables bind_variables_then env_kinds].
        rewrite cc_code_env_kinds, app_nil_r. cbn [rev]. exact Hbc.
      * rewrite so_binding_code in Hbs. exact Hbs.
    + reflexivity.
    + cbn [rebuild_binding_cc].
      destruct (static_const_is_used sol s const); [ | reflexivity ].
      cbn [forallb]. rewrite andb_true_r. apply static_const_ok, Hbs.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en Hc Hs.
    cbn [calls_plain] in Hc. apply andb_true_iff in Hc as [Hhc Hc].
    rewrite so_let_cont in Hs. apply andb_true_iff in Hs as [Hhs Hb].
    rewrite rebuild_cc_let_cont.
    match goal with
    | |- context [ RB ?en1 body ] =>
      assert (IHb : SO (RB en1 body) = true) by (apply IH; assumption)
    end.
    destruct (among k _); [ | exact IHb ].
    rewrite so_let_cont, IHb, andb_true_r. apply IHh; assumption.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en Hc Hs.
    cbn [calls_plain] in Hc. apply andb_true_iff in Hc as [Hhc Hc].
    rewrite so_let_cont_rec in Hs. apply andb_true_iff in Hs as [Hhs Hb].
    rewrite rebuild_cc_let_cont_rec, so_let_cont_rec.
    match goal with
    | |- context [ RB ?en1 body ] => rewrite (IH en1 Hc Hb), andb_true_r
    end.
    rewrite forallb_forall in Hhc, Hhs |- *. intros [k' h'] Hin.
    apply in_map_iff in Hin as [[k h] [Heq Hin]]. injection Heq as <- <-.
    apply (IHh k h Hin); [ exact (Hhc _ Hin) | exact (Hhs _ Hin) ].
  - (* Apply *)
    intros ap en Hc Hs. cbn [calls_plain] in Hc.
    apply andb_true_iff in Hc as [Hr Hk]. cbn [rebuild_expr_cc].
    exact (rebuild_apply_cc_ok en ap Hr Hk Hs).
  - (* Apply_cont *)
    intros ac en _ Hs. cbn [rebuild_expr_cc]. exact (apply_cont_ok en ac Hs).
  - (* Switch *)
    intros scrutinee arms en _ Hs. cbn [simples_ok] in Hs.
    apply andb_true_iff in Hs as [Hsc Harms]. cbn [rebuild_expr_cc].
    exact (switch_ok en scrutinee arms Hsc Harms).
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en dc mask Hc Hs.
    cbn [handler_calls_plain] in Hc. apply andb_true_iff in Hc as [_ Hc].
    rewrite so_handler in Hs. apply andb_true_iff in Hs as [Hk Hb].
    rewrite rebuild_handler_cc_unfold, so_handler,
      kept_handler_metadata_kinds, (HH_filter _ _ Hk).
    destruct (dc && _); [ reflexivity | ]. apply IH; assumption.
Qed.

End Keeps.

(* ======================================================================== *)
(* Erasable, and fresh symbols                                              *)
(* ======================================================================== *)

Section Rebuilt.

Variables (F : facts) (sol : solution) (cannot_change : node -> bool)
  (code_deps : finite_map code_dep) (b : bool) (ucids : list code_id)
  (codes : finite_map code_metadata) (ff : positive).

Hypothesis Hres :
  ∀ (ap : apply_expr) (pd rd : list param_decision),
    apply_calling_convention_change F sol cannot_change code_deps ucids codes
      ap
    = Changing_calling_convention pd rd ->
    forallb kind_not_region (kept_kinds rd) = true.

Abbreviation RB0 :=
  (rebuild_expr_cc F sol cannot_change code_deps b ucids codes ff None
     FlambdaReaperRebuild.empty_env).

Lemma rebuild_cc_keeps_simples (P : simple -> bool) (Q : symbol -> bool)
    (H : list flambda_kind -> bool) (e : expr) :
  (∀ x : variable, P (Var x) = true) ->
  (∀ c : const, const_no_region_poison c = true -> P (Const c) = true) ->
  (∀ l : list flambda_kind, forallb kind_not_region l = true -> H l = true) ->
  (∀ (mask : list bool) (l : list flambda_kind),
     H l = true -> H (filter_kept mask l) = true) ->
  regions_plain [] e = true -> simples_ok P Q H e = true ->
  simples_ok P Q H (RB0 e) = true.
Proof.
  intros HP_var HP_const HH_plain HH_filter Hpl Hs.
  exact (proj1 (rebuild_cc_simples P Q H HP_var HP_const HH_plain HH_filter
                  F sol cannot_change code_deps b ucids codes ff Hres)
           e FlambdaReaperRebuild.empty_env
           (proj1 regions_plain_calls e [] Hpl) Hs).
Qed.

Lemma rebuild_cc_calls_return (e : expr) :
  regions_plain [] e = true -> calls_return_no_region [] (RB0 e) = true.
Proof.
  intros Hpl. apply (proj1 calls_return_simples _ [] conts_plain_nil).
  apply rebuild_cc_keeps_simples; try reflexivity.
  - intros l Hl. exact Hl.
  - intros mask l Hl. apply filter_kept_forallb, Hl.
  - exact Hpl.
  - exact (proj1 calls_plain_handlers e []
             (proj1 regions_plain_calls e [] Hpl)).
Qed.

Lemma rebuild_cc_no_region_poison (e : expr) :
  regions_plain [] e = true -> no_region_poison e = true ->
  no_region_poison (RB0 e) = true.
Proof.
  intros Hpl Hp. rewrite (proj1 no_region_poison_simples) in Hp |- *.
  apply rebuild_cc_keeps_simples; try reflexivity; try assumption.
  intros c Hc. exact Hc.
Qed.

Lemma rebuild_cc_erasable (e : expr) :
  regions_plain [] e = true -> erasable e empty_static_env = true ->
  erasable (RB0 e) empty_static_env = true.
Proof.
  intros Hpl He. unfold erasable in He |- *.
  apply andb_true_iff in He as [He _]. apply andb_true_iff in He as [_ Hp].
  rewrite rebuild_cc_calls_return, rebuild_cc_no_region_poison;
    try assumption.
  reflexivity.
Qed.

Lemma rebuild_cc_expr_below (base : symbol) (e : expr) :
  regions_plain [] e = true -> expr_below base e = true ->
  expr_below base (RB0 e) = true.
Proof.
  intros Hpl Hb. rewrite (proj1 (expr_below_simples base)) in Hb |- *.
  apply rebuild_cc_keeps_simples; try reflexivity; assumption.
Qed.

Lemma rebuild_cc_fresh (base : symbol) (e : expr) :
  regions_plain [] e = true ->
  region_names_fresh base e empty_static_env = true ->
  region_names_fresh base (RB0 e) empty_static_env = true.
Proof.
  intros Hpl Hf. unfold region_names_fresh in Hf |- *.
  apply andb_true_iff in Hf as [Hf _]. apply andb_true_iff in Hf as [Hb _].
  rewrite (rebuild_cc_expr_below base e Hpl Hb). reflexivity.
Qed.

End Rebuilt.

(* ======================================================================== *)
(* The plain rebuild                                                        *)
(* ======================================================================== *)

(* No code changes its calling convention where every code is in the
   table. *)
Lemma all_unchanged (F : facts) (sol : solution)
    (code_deps : finite_map code_dep) (ucids : list code_id)
    (codes : finite_map code_metadata) :
  ∀ (ap : apply_expr) (pd rd : list param_decision),
    apply_calling_convention_change F sol (fun _ => true) code_deps ucids
      codes ap
    = Changing_calling_convention pd rd ->
    forallb kind_not_region (kept_kinds rd) = true.
Proof.
  intros ap pd rd. unfold apply_calling_convention_change.
  destruct (code_id_actually_called F sol ap) as [id | ];
    [ | discriminate ].
  unfold get_calling_convention_change.
  destruct (lookup id codes) as [md | ]; [ | discriminate ].
  unfold code_calling_convention_change, cannot_change_calling_convention.
  rewrite orb_true_r. discriminate.
Qed.

Lemma rebuild_unit_as_cc (F : facts) (T : tables) (b : bool) (e : expr) :
  rebuild_unit_from_tables F T b e
  = rebuild_expr_cc F (solution_of_tables T) (fun _ => true) [] b
      (static_names code_ids_bound e) (bound_code_metadata e)
      (fresh_name (names_of e)) None FlambdaReaperRebuild.empty_env e.
Proof.
  unfold rebuild_unit_from_tables, rebuild_unit.
  symmetry. apply (FlambdaReaperRebuildProofs.rebuild_expr_cc_unchanged
                     F (solution_of_tables T) (fun _ => true) [] b
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     (fresh_name (names_of e)) None (fun _ _ => eq_refl)).
Qed.

Lemma rebuild_keeps_erasable (F : facts) (T : tables) (b : bool) (e : expr) :
  regions_plain [] e = true -> erasable e empty_static_env = true ->
  erasable (rebuild_unit_from_tables F T b e) empty_static_env = true.
Proof.
  intros Hpl He. rewrite rebuild_unit_as_cc.
  apply rebuild_cc_erasable; [ apply all_unchanged | exact Hpl | exact He ].
Qed.

Lemma rebuild_keeps_fresh (F : facts) (T : tables) (b : bool) (e : expr)
    (base : symbol) :
  regions_plain [] e = true ->
  region_names_fresh base e empty_static_env = true ->
  region_names_fresh base (rebuild_unit_from_tables F T b e) empty_static_env
  = true.
Proof.
  intros Hpl Hf. rewrite rebuild_unit_as_cc.
  apply rebuild_cc_fresh; [ apply all_unchanged | exact Hpl | exact Hf ].
Qed.
