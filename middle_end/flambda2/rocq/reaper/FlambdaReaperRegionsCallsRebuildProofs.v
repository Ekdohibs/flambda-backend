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

(* The commutation of the rebuild with unboxing
   (FlambdaReaperUnboxing.rebuild_expr_u, in either mode, with no change
   of representation) with the erasure of the regions, on the units of
   FlambdaReaperRegionsCallsSpec.regions_calls: those of
   FlambdaReaperRegionsUnboxRebuildProofs.v, case for case, with the
   calls, the code and the sets of closures of
   FlambdaReaperRegionsCcRebuildProofs.v. A call is rebuilt alike in the
   two rebuilds (its parts read the code deps, whose nodes are not the
   constants, and facts that the erasure keeps), and its wrapper has
   parameters of the kinds of the results of the call and of the leaves
   of unboxed fields, none of the kind [Region], so it is erased to
   itself; code is erased in its body alone.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsCallsProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaRegionsSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsCallsSpec
  FlambdaReaperRegionsRebuildProofs FlambdaReaperRegionsTraverseProofs
  FlambdaReaperRegionsUnboxRebuildProofs.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaReaperFlowProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperUnboxingRefinesCallsProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module RCP := FlambdaReaperUnboxingRefinesCallsProofs.

(* ======================================================================== *)
(* The parameters of a wrapper                                              *)
(* ======================================================================== *)

Lemma leaves_paths :
  ∀ (u : unboxed_field) (k : flambda_kind),
    map snd (unboxed_field_leaves k u) = map snd (ufield_paths k u).
Proof.
  refine (URP.unboxed_field_rec _ _ _).
  - intros v k. reflexivity.
  - intros fields IH k.
    induction fields as [ | [f u] fields IHl]; [ reflexivity | ].
    change (unboxed_field_leaves k (Unboxed ((f, u) :: fields)))
      with (unboxed_field_leaves (field_kind f) u
            ++ unboxed_field_leaves k (Unboxed fields)).
    change (ufield_paths k (Unboxed ((f, u) :: fields)))
      with (map (fun pk => (f :: fst pk, snd pk))
              (ufield_paths (field_kind f) u)
            ++ ufield_paths k (Unboxed fields)).
    rewrite !map_app. f_equal.
    + rewrite map_map, (IH f u (or_introl eq_refl)).
      apply map_ext. intros [p kk]. reflexivity.
    + apply IHl. intros f' u' Hin. apply (IH f' u'). right. exact Hin.
Qed.

Lemma rename_leaves_kinds (u : unboxed_field) (k : flambda_kind)
    (n : positive) :
  map snd (unboxed_field_leaves k (fst (rename_unboxed_field u n)))
  = map snd (unboxed_field_leaves k u).
Proof.
  rewrite !leaves_paths.
  destruct (RCP.rename_ok u k n) as (_ & Hp & _). rewrite Hp. reflexivity.
Qed.

Lemma wrapper_params_cons (wd : wrapper_decision) (k : flambda_kind)
    (l : list (wrapper_decision * flambda_kind)) :
  wrapper_params ((wd, k) :: l)
  = match wd with
    | Wrapper_keep x => [(x, k)]
    | Wrapper_delete => []
    | Wrapper_unbox fields => fields_leaves fields
    end
    ++ wrapper_params l.
Proof. reflexivity. Qed.

(* The decisions of the callee for the results of a call, with the kinds
   of the results: none of the kind [Region], and no unboxed field of that
   kind. *)
Definition fd_plain (fd : list (decision * flambda_kind)) : Prop :=
  ∀ (d : decision) (k : flambda_kind),
    (d, k) ∈ fd ->
    kind_not_region k = true
    ∧ ∀ fields : unboxed_fields,
        d = Unbox fields ->
        forallb kind_not_region (map snd (fields_leaves fields)) = true.

Lemma wrapper_params_plain (fd : list (decision * flambda_kind)) :
  ∀ n : positive,
    fd_plain fd ->
    forallb kind_not_region
      (map snd (wrapper_params (freshen_decisions fd n)))
    = true.
Proof.
  induction fd as [ | [d k] fd IH]; intros n H; [ reflexivity | ].
  assert (Hrest : fd_plain fd)
    by (intros d' k' Hin; apply H; right; exact Hin).
  destruct (H d k (or_introl eq_refl)) as [Hk Hf].
  destruct d as [[kk | ] | fields | kk]; cbn [freshen_decisions].
  - rewrite wrapper_params_cons. cbn [app map forallb snd].
    rewrite Hk. apply IH, Hrest.
  - rewrite wrapper_params_cons. cbn [app]. apply IH, Hrest.
  - destruct (rename_unboxed_field (Unboxed fields) n) as [[v | fields'] n']
      eqn:Hr.
    + rewrite wrapper_params_cons. cbn [app]. apply IH, Hrest.
    + rewrite wrapper_params_cons, map_app, forallb_app.
      pose proof (rename_leaves_kinds (Unboxed fields) Value n) as Hl.
      rewrite Hr in Hl. cbn [fst] in Hl.
      change (unboxed_field_leaves Value (Unboxed fields'))
        with (fields_leaves fields') in Hl.
      change (unboxed_field_leaves Value (Unboxed fields))
        with (fields_leaves fields) in Hl.
      rewrite Hl, (Hf fields eq_refl). apply IH, Hrest.
  - rewrite wrapper_params_cons. cbn [app map forallb snd].
    rewrite Hk. apply IH, Hrest.
Qed.

(* What [make_apply_wrapper_u] makes is erased to itself. *)
Lemma erase_wrapper (fx : bool) (next : positive)
    (subk : list param_subkind) (en : unbox_env) (ap : apply_expr)
    (fd : list (decision * flambda_kind)) (nr : bool) :
  fd_plain fd ->
  erase [] (make_apply_wrapper_u fx next None subk en ap fd nr)
  = make_apply_wrapper_u fx next None subk en ap fd nr.
Proof.
  intros Hfd. pose proof (wrapper_params_plain fd next Hfd) as Hp.
  unfold make_apply_wrapper_u. cbv beta iota zeta.
  destruct (fx && nr).
  - cbn [erase]. rewrite erase_handler_no_region by exact Hp. reflexivity.
  - match goal with
    | |- context [ wrapper_args_u ?a ?b ?c ] =>
      destruct (wrapper_args_u a b c) as [args | ]
    end.
    + match goal with
      | |- context [ args_are_params ?a ?b ] => destruct (args_are_params a b)
      end; [ reflexivity | ].
      cbn [erase]. rewrite erase_handler_no_region by exact Hp.
      cbn [erase]. rewrite erase_apply_cont_nil. reflexivity.
    + cbn [erase]. rewrite erase_handler_no_region by exact Hp.
      reflexivity.
Qed.

(* A closure of a static set keeps its representation. *)
Lemma static_set_repr_nil (s : symbol) (const : static_const) :
  static_set_repr [] s const = None.
Proof.
  destruct const; try reflexivity. cbn [static_set_repr].
  match goal with
  | |- fold_right _ None ?l = None =>
    induction l as [ | [fs sym] l IH]; [ reflexivity | exact IH ]
  end.
Qed.

(* ======================================================================== *)
(* The parts of the two rebuilds                                            *)
(* ======================================================================== *)

Section CallsParts.

Variable fx : bool.
Variables F F' : facts.
Variables sol sol' : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable b : bool.
Variable ucids : list code_id.
Variable codes : finite_map code_metadata.
Variable ub : list (node * unboxed_fields).
Variable next : positive.

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
(* The nodes of the code deps are not the constants. *)
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep), lookup id cds = Some cd -> cd_ok cd.
(* No variable of an unboxed field is of the kind [Region]. *)
Hypothesis Hleaves : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub -> forallb kind_not_region (map snd (fields_leaves fields))
                      = true.

Lemma node_used_eq (n : node) (k : flambda_kind) :
  n <> Node_all_constants -> is_node_used sol' n k = is_node_used sol n k.
Proof.
  intros Hn. unfold is_node_used.
  destruct k; try reflexivity; rewrite (Hu n Hn); reflexivity.
Qed.

Lemma decisions_u_eq (nodes : list node) (kinds : list flambda_kind) :
  Forall not_consts nodes ->
  decisions_u sol' ub nodes kinds = decisions_u sol ub nodes kinds.
Proof.
  intros Hn. unfold decisions_u. apply map_ext_in. intros [n k] Hin.
  destruct (unboxed_of_node ub n); [ reflexivity | ].
  rewrite (node_used_eq n k
             (proj1 (Forall_forall _ _) Hn n (in_combine_l _ _ _ _ Hin))).
  reflexivity.
Qed.

Lemma fixed_arity_decisions_eq (nodes : list node)
    (kinds : list flambda_kind) :
  Forall not_consts nodes ->
  fixed_arity_decisions sol' ub nodes kinds
  = fixed_arity_decisions sol ub nodes kinds.
Proof.
  intros Hn. unfold fixed_arity_decisions. apply map_ext_in.
  intros [n k] Hin. destruct (unboxed_of_node ub n); [ reflexivity | ].
  rewrite (node_used_eq n k
             (proj1 (Forall_forall _ _) Hn n (in_combine_l _ _ _ _ Hin))).
  reflexivity.
Qed.

Lemma code_change_u_eq (id : code_id) (md : code_metadata) :
  code_change_u sol' cc cds ucids ub id md
  = code_change_u sol cc cds ucids ub id md.
Proof.
  unfold code_change_u.
  destruct (cannot_change_calling_convention cc ucids id); [ reflexivity | ].
  destruct (lookup id cds) as [cd | ] eqn:Hl; [ | reflexivity ].
  destruct (Hcds id cd Hl) as (C1 & _ & C3 & _).
  rewrite (decisions_u_eq _ _ C1), (decisions_u_eq _ _ C3). reflexivity.
Qed.

Lemma code_continuation_decisions_eq (id : code_id) (md : code_metadata)
    (ch : calling_convention_change_u) (k k' : continuation) :
  code_continuation_decisions sol' cds ub id md ch k k'
  = code_continuation_decisions sol cds ub id md ch k k'.
Proof.
  unfold code_continuation_decisions.
  destruct (lookup id cds) as [cd | ] eqn:Hl; [ | reflexivity ].
  destruct (Hcds id cd Hl) as (_ & _ & C3 & C4 & _).
  rewrite (fixed_arity_decisions_eq _ _ C3).
  rewrite (fixed_arity_decisions_eq [cd_exn cd] [Value]);
    [ reflexivity | constructor; [ exact C4 | constructor ] ].
Qed.

Lemma called_eq (ap : apply_expr) :
  code_id_actually_called F' sol' ap = code_id_actually_called F sol ap.
Proof.
  unfold code_id_actually_called.
  rewrite (code_id_actually_directly_called_eq F F' sol sol' HC HA Hsrc Has
             Hss).
  reflexivity.
Qed.

Lemma apply_change_u_eq (ap : apply_expr) :
  apply_change_u F' sol' cc cds ucids codes ub ap
  = apply_change_u F sol cc cds ucids codes ub ap.
Proof.
  unfold apply_change_u. rewrite called_eq.
  destruct (code_id_actually_called F sol ap) as [id | ]; [ | reflexivity ].
  destruct (lookup id codes); [ apply code_change_u_eq | reflexivity ].
Qed.

Lemma apply_my_closure_u_eq (ap : apply_expr) :
  apply_my_closure_u F' sol' cc cds ucids codes ub ap
  = apply_my_closure_u F sol cc cds ucids codes ub ap.
Proof. unfold apply_my_closure_u. rewrite called_eq. reflexivity. Qed.

Lemma result_has_source_of_eq (ws : node -> bool) (i : nat) :
  result_has_source_of F' sol' ws i = result_has_source_of F sol ws i.
Proof. unfold result_has_source_of. rewrite HC, Hsrc. reflexivity. Qed.

Lemma never_returns_of_call_eq (ep : closure_entry_point) (nres : nat)
    (num_args : list nat) :
  ∀ src : allocation_sources,
    never_returns_of_call F' sol' ep src num_args nres
    = never_returns_of_call F sol ep src num_args nres.
Proof.
  induction num_args as [ | n rest IH]; intros src; [ reflexivity | ].
  cbn [never_returns_of_call]. destruct src as [ | cs]; [ reflexivity | ].
  rewrite (get_field_sources_eq F F' sol sol' HC Hsrc Has Hss).
  destruct (get_field_sources F sol cs (Field_call_witness ep)) as [ | ws];
    [ reflexivity | ].
  destruct rest as [ | r rs].
  - apply existsb_ext_in. intros i _. rewrite result_has_source_of_eq.
    reflexivity.
  - rewrite (get_field_sources_eq F F' sol sol' HC Hsrc Has Hss). apply IH.
Qed.

Lemma apply_never_returns_eq (ap : apply_expr) :
  apply_never_returns F' sol' cds codes ap
  = apply_never_returns F sol cds codes ap.
Proof.
  unfold apply_never_returns. rewrite called_eq.
  destruct (code_id_actually_called F sol ap) as [id | ].
  - destruct (lookup id codes); [ | reflexivity ].
    unfold code_never_returns. rewrite Hsrc. reflexivity.
  - destruct (ap_callee ap); try reflexivity;
      rewrite (get_direct_sources_eq sol sol' Hsrc Has Hss);
      destruct (ap_call_kind ap); apply never_returns_of_call_eq.
Qed.

Lemma kept_code_mask_eq (ap : apply_expr) :
  kept_code_mask F' sol' cds codes ap = kept_code_mask F sol cds codes ap.
Proof.
  unfold kept_code_mask. rewrite called_eq.
  destruct (code_id_actually_called F sol ap) as [id | ]; [ | reflexivity ].
  destruct (lookup id codes); [ | reflexivity ].
  destruct (lookup id cds) as [cd | ] eqn:Hl; [ | reflexivity ].
  destruct (Hcds id cd Hl) as (C1 & _). f_equal. apply map_ext_in.
  intros n Hn. apply Hu. exact (proj1 (Forall_forall _ _) C1 n Hn).
Qed.

(* The code that a call of code that changes its calling convention is
   known to call unboxes its results as the unboxed fields say. *)
Lemma apply_change_unbox (ap : apply_expr) (pd rd : list decision) :
  apply_change_u F sol cc cds ucids codes ub ap = Changing_u pd rd ->
  ∀ fields : unboxed_fields, Unbox fields ∈ rd -> ∃ n, (n, fields) ∈ ub.
Proof.
  unfold apply_change_u.
  destruct (code_id_actually_called F sol ap) as [id | ]; [ | discriminate ].
  destruct (lookup id codes) as [md | ]; [ | discriminate ].
  unfold code_change_u.
  destruct (cannot_change_calling_convention cc ucids id);
    [ discriminate | ].
  destruct (lookup id cds) as [cd | ]; [ | discriminate ].
  intros Heq. injection Heq as _ Hrd. subst rd. intros fields Hin.
  unfold decisions_u in Hin. apply in_map_iff in Hin as ([n k] & Heq & _).
  destruct (unboxed_of_node ub n) as [fs | ] eqn:Hu1.
  - injection Heq as Hfs. subst fs. exists n.
    exact (keyed_lookup_in node_eqb URP.node_eqb_decides _ _ _ Hu1).
  - destruct (is_node_used sol n k); discriminate Heq.
Qed.

Section Env.

Variables en en' : unbox_env.
Hypothesis Hen : uenv_rel en en'.

Lemma kept_arguments_u_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds (ue_env en))) (ap_args ap) = true ->
  kept_arguments_u fx F' sol' cds codes ub en' ap
  = kept_arguments_u fx F sol cds codes ub en ap.
Proof.
  intros H. destruct Hen as (He & _ & _). unfold kept_arguments_u.
  rewrite kept_code_mask_eq.
  destruct (if fx then kept_code_mask F sol cds codes ap else None)
    as [mask | ].
  - exact (mask_arguments_eq _ _ He _ _ H).
  - exact (rebuild_arguments_eq F F' (solution_without_unboxed sol ub)
             (solution_without_unboxed sol' ub) HC HP Hpac
             (hu_u sol sol' ub Hu) Hsrc Has Hss _ _ He ap H).
Qed.

Lemma get_extra_args_eq (ds : list decision)
    (args : list (simple * flambda_kind)) :
  forallb (simple_not_region (env_kinds (ue_env en))) (map fst args)
  = true ->
  get_extra_args sol' ub (ue_env en') ds args
  = get_extra_args sol ub (ue_env en) ds args.
Proof.
  destruct Hen as (He & _ & _). revert ds.
  induction args as [ | [s k] args IH]; intros ds H;
    [ destruct ds; reflexivity | ].
  cbn [map forallb fst] in H. apply andb_true_iff in H as [Hs H].
  pose proof (rewrite_simple_eq _ _ (hu_u sol sol' ub Hu) _ _ He s Hs) as Hr.
  destruct ds as [ | d ds]; cbn [get_extra_args].
  - rewrite (IH [] H), Hr. reflexivity.
  - rewrite (IH ds H), Hr. reflexivity.
Qed.

Lemma rebuild_exn_extra_args_u_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds (ue_env en))) (exn_extra_simples ap)
  = true ->
  rebuild_exn_extra_args_u sol' ub en' ap
  = rebuild_exn_extra_args_u sol ub en ap.
Proof.
  intros H. pose proof Hen as (_ & Hd & _). unfold rebuild_exn_extra_args_u.
  rewrite Hd. apply get_extra_args_eq. exact H.
Qed.

Lemma get_args_with_kinds_u_eq (ds : list decision) (args : list simple) :
  forallb (simple_not_region (env_kinds (ue_env en))) args = true ->
  get_args_with_kinds_u sol' ub (ue_env en') ds args
  = get_args_with_kinds_u sol ub (ue_env en) ds args.
Proof.
  destruct Hen as (He & _ & _). revert ds.
  induction args as [ | a args IH]; intros ds H;
    [ destruct ds; reflexivity | ].
  cbn [forallb] in H. apply andb_true_iff in H as [Ha H].
  pose proof (rewrite_simple_eq _ _ (hu_u sol sol' ub Hu) _ _ He a Ha) as Hr.
  destruct ds as [ | d ds]; [ reflexivity | ]. cbn [get_args_with_kinds_u].
  rewrite (IH ds H), Hr. reflexivity.
Qed.

Lemma groups_u_eq (a : arity) (args : list simple) (pd : list decision) :
  forallb (simple_not_region (env_kinds (ue_env en))) args = true ->
  map (fun '(args, decisions) =>
         get_args_with_kinds_u sol' ub (ue_env en') decisions args)
    (combine (group_by_parameter a args) (group_by_parameter a pd))
  = map (fun '(args, decisions) =>
           get_args_with_kinds_u sol ub (ue_env en) decisions args)
      (combine (group_by_parameter a args) (group_by_parameter a pd)).
Proof.
  intros H. apply map_ext_in. intros [g ds] Hin. apply get_args_with_kinds_u_eq.
  apply in_combine_l in Hin. rewrite forallb_forall in H |- *.
  intros x Hx. apply H. unfold group_by_parameter in Hin.
  exact (group_by_sizes_in _ _ _ _ Hin Hx).
Qed.

Lemma make_apply_wrapper_u_eq (subk : list param_subkind) (ap : apply_expr)
    (fd : list (decision * flambda_kind)) (nr : bool) :
  make_apply_wrapper_u fx next None subk en' ap fd nr
  = make_apply_wrapper_u fx next None subk en ap fd nr.
Proof.
  destruct Hen as (_ & Hd & _). unfold make_apply_wrapper_u. rewrite Hd.
  reflexivity.
Qed.

Lemma rebuild_apply_u_eq (ap : apply_expr) :
  forallb (simple_not_region (env_kinds (ue_env en)))
    (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) = true ->
  rebuild_apply_u fx F' sol' cc cds b ucids codes ub next None en' ap
  = rebuild_apply_u fx F sol cc cds b ucids codes ub next None en ap.
Proof.
  intros H. pose proof Hen as (He & Hd & _). cbn [forallb] in H.
  apply andb_true_iff in H as [Hc H].
  rewrite forallb_app in H. apply andb_true_iff in H as [Hargs Hextra].
  unfold rebuild_apply_u.
  rewrite (simple_is_dead_eq sol sol' Hsrc _ _ He _ Hc).
  rewrite (existsb_ext_in (simple_is_dead sol' (ue_env en'))
             (simple_is_dead sol (ue_env en)))
    by (intros s Hs; rewrite forallb_forall in Hargs;
        exact (simple_is_dead_eq sol sol' Hsrc _ _ He s (Hargs s Hs))).
  destruct (_ || _); [ reflexivity | ].
  rewrite (rebuild_exn_extra_args_u_eq ap Hextra).
  destruct (rebuild_exn_extra_args_u sol ub en ap) as [extra | ];
    [ | reflexivity ].
  rewrite apply_change_u_eq.
  destruct (apply_change_u F sol cc cds ucids codes ub ap) as [ | pd rd].
  - destruct (unboxed_of ub (ap_callee ap)); [ reflexivity | ].
    rewrite (rewrite_call_kind_eq F F' sol sol' b ucids HC HA Hsrc Has Hss),
      (kept_arguments_u_eq ap Hargs), apply_never_returns_eq.
    apply make_apply_wrapper_u_eq.
  - rewrite apply_my_closure_u_eq, (groups_u_eq _ _ _ Hargs),
      (rewrite_simple_eq _ _ (hu_u sol sol' ub Hu) _ _ He _ Hc),
      (rewrite_call_kind_eq F F' sol sol' b ucids HC HA Hsrc Has Hss),
      apply_never_returns_eq.
    destruct (match apply_my_closure_u F sol cc cds ucids codes ub ap with
              | Some fields => unboxed_args ub fields (ap_callee ap)
              | None => Some []
              end) as [fc | ]; [ | reflexivity ].
    match goal with
    | |- context [ all_some ?l ] => destruct (all_some l) as [gs | ]
    end; [ | reflexivity ].
    apply make_apply_wrapper_u_eq.
Qed.

End Env.

(* What the rebuild makes of a call is erased to itself. *)
Lemma erase_rebuild_apply_u (en : unbox_env) (ap : apply_expr) :
  forallb kind_not_region (unarize (ap_return_arity ap)) = true ->
  erase [] (rebuild_apply_u fx F sol cc cds b ucids codes ub next None en ap)
  = rebuild_apply_u fx F sol cc cds b ucids codes ub next None en ap.
Proof.
  intros Hr. unfold rebuild_apply_u.
  destruct (_ || _); [ reflexivity | ].
  destruct (rebuild_exn_extra_args_u sol ub en ap) as [extra | ];
    [ | reflexivity ].
  destruct (apply_change_u F sol cc cds ucids codes ub ap) as [ | pd rd]
    eqn:Hch.
  - destruct (unboxed_of ub (ap_callee ap)); [ reflexivity | ].
    apply erase_wrapper. intros d k Hin.
    apply in_map_iff in Hin as (kk & Heq & Hkk). injection Heq as <- <-.
    split; [ rewrite forallb_forall in Hr; exact (Hr _ Hkk) | ].
    intros fields Hf. discriminate Hf.
  - destruct (match apply_my_closure_u F sol cc cds ucids codes ub ap with
              | Some fields => unboxed_args ub fields (ap_callee ap)
              | None => Some []
              end) as [fc | ]; [ | reflexivity ].
    match goal with
    | |- context [ all_some ?l ] => destruct (all_some l) as [gs | ]
    end; [ | reflexivity ].
    apply erase_wrapper. intros d k Hin. split.
    + apply in_combine_r in Hin. rewrite forallb_forall in Hr. exact (Hr k Hin).
    + intros fields ->. apply in_combine_l in Hin.
      destruct (apply_change_unbox ap pd rd Hch fields Hin) as [n Hn].
      exact (Hleaves n fields Hn).
Qed.

End CallsParts.

(* ======================================================================== *)
(* The steps of the rebuild with unboxing                                   *)
(* ======================================================================== *)

Section Unfold.

Variable fx : bool.
Variables (F0 : facts) (s0 : solution) (cc0 : node -> bool)
  (cds0 : finite_map code_dep) (b0 : bool) (u0 : list code_id)
  (c0 : finite_map code_metadata) (ub0 : list (node * unboxed_fields))
  (n0 : positive).

Abbreviation RU0 := (rebuild_expr_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 []
                       None).

(* The [Let] of a set of closures that is not unboxed and is followed by
   no projection. *)
Lemma ru_let_set (en : unbox_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  is_pending_projection (ue_env en) x d = false -> named_is_set d = true ->
  projections_of_set x d body = [] ->
  is_unboxed_name ub0 (Node_variable x) = false ->
  RU0 en (Let x kind d body)
  = if is_dead_var s0 x kind then Invalid
    else if is_var_used s0 x kind
    then
      Let x kind (rebuild_set_u F0 s0 ub0 [] (ue_env en) x d)
        (RU0 (ubind_variables en [(x, kind)]) body)
    else RU0 (ubind_variables en [(x, kind)]) body.
Proof.
  intros Hp Hs Hpr Hn. cbn [rebuild_expr_u]. rewrite Hp.
  destruct d; try discriminate Hs; rewrite Hpr; cbn [existsb];
    rewrite Hn, !orb_false_r; reflexivity.
Qed.

Lemma rb_code (en : unbox_env) (id : code_id) (c : code) :
  rebuild_binding_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 [] None en
    (Bind_code id c)
  = if is_code_id_used s0 id
    then [Bind_code id (rebuild_code_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 []
                          None id c)]
    else [Bind_deleted_code id].
Proof. reflexivity. Qed.

Lemma rb_symbol (en : unbox_env) (s : symbol) (const : static_const) :
  rebuild_binding_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 [] None en
    (Bind_symbol s const)
  = if static_const_is_used s0 s const
    then [Bind_symbol s (rebuild_static_const (solution_without_unboxed s0 ub0)
                           (ue_env en) s const)]
    else [].
Proof. cbn [rebuild_binding_u]. rewrite static_set_repr_nil. reflexivity. Qed.

End Unfold.

Lemma rebuild_set_u_nil (F0 : facts) (s0 : solution)
    (ub0 : list (node * unboxed_fields)) (en : rebuild_env) (x : variable)
    (d : named) :
  rebuild_set_u F0 s0 ub0 [] en x d
  = rebuild_set (solution_without_unboxed s0 ub0) en
      (set_closure_nodes F0 (Node_variable x)) d.
Proof. reflexivity. Qed.

Lemma set_binders_let_set (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  named_is_set d = true ->
  set_binders (Let x kind d body) = x :: set_binders body.
Proof. destruct d; try discriminate; reflexivity. Qed.

Lemma set_binders_let_other (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  named_is_set d = false ->
  set_binders (Let x kind d body) = set_binders body.
Proof. destruct d; try discriminate; reflexivity. Qed.

Lemma not_unboxed_name (ub : list (node * unboxed_fields)) (x : variable) :
  unboxed_of_node ub (Node_variable x) = None ->
  is_unboxed_name ub (Node_variable x) = false.
Proof.
  intros H. unfold is_unboxed_name, Keyed.is_bound.
  unfold unboxed_of_node in H. rewrite H. reflexivity.
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Section CallsCommute.

Variable fx : bool.
Variables F F' : facts.
Variables sol sol' : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable b : bool.
Variable ucids : list code_id.
Variable codes : finite_map code_metadata.
Variable ub : list (node * unboxed_fields).
Variable next : positive.

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
Hypothesis Hcds :
  ∀ (id : code_id) (cd : code_dep), lookup id cds = Some cd -> cd_ok cd.
Hypothesis Hleaves : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub -> forallb kind_not_region (map snd (fields_leaves fields))
                      = true.

Abbreviation RU := (rebuild_expr_u fx F sol cc cds b ucids codes ub next []
                      None).
Abbreviation RU' := (rebuild_expr_u fx F' sol' cc cds b ucids codes ub next
                       [] None).
Abbreviation RH := (rebuild_handler_u fx F sol cc cds b ucids codes ub next
                      [] None).
Abbreviation RH' := (rebuild_handler_u fx F' sol' cc cds b ucids codes ub
                       next [] None).

Lemma rebuild_set_u_eq (en en' : rebuild_env) (x : variable) (d : named) :
  env_rel en en' ->
  forallb (simple_not_region (env_kinds en)) (simples_of_named d) = true ->
  rebuild_set_u F' sol' ub [] en' x d = rebuild_set_u F sol ub [] en x d.
Proof.
  intros Hen Hd. rewrite !rebuild_set_u_nil.
  unfold set_closure_nodes. rewrite HC.
  exact (rebuild_set_eq _ _ (hu_u sol sol' ub Hu) Hfu _ _ Hen _ _ Hd).
Qed.

Lemma rebuild_u_calls_erase_commutes :
  (∀ e : expr, ∀ en en' : unbox_env,
     regions_plain (env_kinds (ue_env en)) e = true ->
     sets_alone e = true ->
     begins_kept ub (region_binders e) ->
     begins_kept ub (set_binders e) ->
     uenv_rel en en' -> uenv_plain en ->
     erase [] (RU en e) = RU' en' (erase [] e))
  ∧ (∀ h : continuation_handler,
       ∀ (en en' : unbox_env) (dc : bool) (ds : list decision),
         handler_regions_plain (env_kinds (ue_env en)) h = true ->
         handler_sets_alone h = true ->
         begins_kept ub (handler_region_binders h) ->
         begins_kept ub (handler_set_binders h) ->
         forallb decision_plain ds = true ->
         uenv_rel en en' ->
         erase_handler [] (RH en dc ds h)
         = RH' en' dc ds (erase_handler [] h)).
Proof.
  apply regions_ind.
  - (* Let *)
    intros x kind d body IH IH2 en en' Hpl Hsa Hbk Hsk Hen Hpe.
    cbn [sets_alone] in Hsa. apply andb_true_iff in Hsa as [Hproj Hsb].
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      assert (Hsk' : begins_kept ub (set_binders body))
        by (intros r Hr; apply Hsk; exact Hr).
      rewrite (ru_let_other fx F sol cc cds b ucids codes ub next en x kind
                 (Prim (Region_primitive p) args) body
                 (pending_plain en x _ Hpe) eq_refl).
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl.
      * (* Begin_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        assert (Hx : unboxed_of_node ub (Node_variable x) = None)
          by (apply Hbk; cbn [region_binders]; left; reflexivity).
        cbn [is_dead_var is_var_used]. rewrite Hx.
        unfold rebuild_let_u.
        cbn [field_read rebuild_named map erase erase_region_primitive].
        apply IH; [ exact Hb | exact Hsb
                  | intros r Hr; apply Hbk; cbn [region_binders]; right;
                    exact Hr
                  | exact Hsk'
                  | apply uenv_rel_region; assumption
                  | apply uenv_plain_bind ].
      * (* Begin_try_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        assert (Hx : unboxed_of_node ub (Node_variable x) = None)
          by (apply Hbk; cbn [region_binders]; left; reflexivity).
        cbn [is_dead_var is_var_used]. rewrite Hx.
        unfold rebuild_let_u.
        cbn [field_read rebuild_named map erase erase_region_primitive].
        apply IH; [ exact Hb | exact Hsb
                  | intros r Hr; apply Hbk; cbn [region_binders]; right;
                    exact Hr
                  | exact Hsk'
                  | apply uenv_rel_region; assumption
                  | apply uenv_plain_bind ].
      * (* End_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        assert (Hbk' : begins_kept ub (region_binders body))
          by (intros r Hr; apply Hbk, region_binders_let, Hr).
        cbn [erase erase_region_primitive].
        rewrite (ru_let_other fx F' sol' cc cds b ucids codes ub next en' x
                   kind (Simple unit_simple) (erase [] body)
                   (pending_plain_rel en en' x _ Hen Hpe) eq_refl).
        rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- destruct (unboxed_of_node ub (Node_variable x)) as [tb | ];
             [ reflexivity | ].
           unfold rebuild_let_u.
           cbn [field_read rebuild_named map erase erase_region_primitive
                  rewrite_simple unit_simple].
           f_equal. apply IH; [ exact Hb | exact Hsb | exact Hbk' | exact Hsk'
                              | apply uenv_rel_bind, Hen
                              | apply uenv_plain_bind ].
        -- apply IH; [ exact Hb | exact Hsb | exact Hbk' | exact Hsk'
                     | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
      * (* End_try_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        assert (Hbk' : begins_kept ub (region_binders body))
          by (intros r Hr; apply Hbk, region_binders_let, Hr).
        cbn [erase erase_region_primitive].
        rewrite (ru_let_other fx F' sol' cc cds b ucids codes ub next en' x
                   kind (Prim (Unary (Opaque_identity Value)) [unit_simple])
                   (erase [] body) (pending_plain_rel en en' x _ Hen Hpe)
                   eq_refl).
        rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- destruct (unboxed_of_node ub (Node_variable x)) as [tb | ];
             [ reflexivity | ].
           unfold rebuild_let_u.
           cbn [field_read rebuild_named map erase erase_region_primitive
                  rewrite_simple unit_simple].
           f_equal. apply IH; [ exact Hb | exact Hsb | exact Hbk' | exact Hsk'
                              | apply uenv_rel_bind, Hen
                              | apply uenv_plain_bind ].
        -- apply IH; [ exact Hb | exact Hsb | exact Hbk' | exact Hsk'
                     | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
    + (* Not a primitive of regions. *)
      rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [Hpl Hla].
      apply andb_true_iff in Hpl as [Hpl Hsetp].
      apply andb_true_iff in Hpl as [Hk Hd].
      assert (Hbk' : begins_kept ub (region_binders body))
        by (intros r Hr; apply Hbk, region_binders_let, Hr).
      rewrite (erase_let_plain _ _ _ _ _ Hrp).
      destruct (named_is_set d) eqn:Hset.
      * (* A set of closures. *)
        rewrite (set_binders_let_set _ _ _ _ Hset) in Hsk.
        assert (Hx : unboxed_of_node ub (Node_variable x) = None)
          by (apply Hsk; left; reflexivity).
        assert (Hsk' : begins_kept ub (set_binders body))
          by (intros r Hr; apply Hsk; right; exact Hr).
        assert (Hprojs : projections_of_set x d body = [])
          by (destruct (projections_of_set x d body); [ reflexivity
                                                      | discriminate Hproj ]).
        rewrite (ru_let_set fx F sol cc cds b ucids codes ub next en x kind d
                   body (pending_plain en x d Hpe) Hset Hprojs
                   (not_unboxed_name ub x Hx)).
        destruct (is_dead_var sol x kind) eqn:Hdead.
        -- rewrite (ru_let_set fx F' sol' cc cds b ucids codes ub next en' x
                      kind d (erase [] body)
                      (pending_plain_rel en en' x d Hen Hpe) Hset
                      (eq_trans (projections_of_set_erase x d body Hsetp)
                         Hprojs)
                      (not_unboxed_name ub x Hx)).
           rewrite (dead_var_eq sol sol' Hsrc), Hdead. reflexivity.
        -- unfold la_check in Hla.
           destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
           ++ (* A local allocation follows. *)
              apply la_body_some in Hlab. subst body.
              apply andb_true_iff in Hla as [Hla Hb'].
              apply andb_true_iff in Hla as [Hla Hargs].
              apply andb_true_iff in Hla as [Hla Hk'].
              apply andb_true_iff in Hla as [_ Hy].
              apply Pos.eqb_eq in Hy. subst y.
              apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
              destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
              apply Pos.eqb_eq in Hargs. subst x'.
              cbn [sets_alone projections_of_set] in Hsb.
              rewrite andb_true_l in Hsb.
              assert (Hbk'' : begins_kept ub (region_binders body'))
                by (intros r Hr; apply Hbk'; exact Hr).
              assert (Hsk'' : begins_kept ub (set_binders body'))
                by (intros r Hr; apply Hsk'; exact Hr).
              cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
              rewrite (ru_let_set fx F' sol' cc cds b ucids codes ub next en'
                         x kind d (erase [] body')
                         (pending_plain_rel en en' x d Hen Hpe) Hset
                         (set_plain_region_head _ _ _ _ _ _ _ Hsetp _)
                         (not_unboxed_name ub x Hx)).
              rewrite (dead_var_eq sol sol' Hsrc), Hdead,
                (var_used_eq sol sol' Hu).
              set (en1 := ubind_variables en [(x, kind)]).
              assert (IHb : erase [] (RU (ubind_variables en1 [(x, kind)])
                                        body')
                            = RU' (ubind_variables en' [(x, kind)])
                                (erase [] body')).
              { apply IH2 with (y := x) (kind' := kind)
                  (d' := Prim (Region_primitive Local_allocation)
                           (Var x :: rs));
                  [ reflexivity | exact Hb' | exact Hsb | exact Hbk''
                  | exact Hsk'' | apply uenv_rel_twice, Hen
                  | apply uenv_plain_bind ]. }
              assert (Hinner :
                erase []
                  (RU en1 (Let x kind
                             (Prim (Region_primitive Local_allocation)
                                (Var x :: rs)) body'))
                = RU' (ubind_variables en' [(x, kind)]) (erase [] body')).
              { rewrite (ru_let_other fx F sol cc cds b ucids codes ub next
                           en1 x kind
                           (Prim (Region_primitive Local_allocation)
                              (Var x :: rs))
                           body' (pending_plain en1 x _ (uenv_plain_bind _ _))
                           eq_refl).
                rewrite Hdead.
                destruct (is_var_used sol x kind) eqn:Hused; [ | exact IHb ].
                rewrite Hx. unfold rebuild_let_u.
                cbn [field_read rebuild_named map].
                change (ue_env en1)
                  with (bind_variables (ue_env en) [(x, kind)]).
                rewrite (rewrite_bound_used sol ub (ue_env en) x kind Hused
                           Hx).
                cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
                exact IHb. }
              destruct (is_var_used sol x kind).
              ** rewrite (erase_let_plain _ _ _ _ _).
                 2: { rewrite rebuild_set_u_nil.
                      apply rebuild_set_plain, Hrp. }
                 rewrite Hinner, (rebuild_set_u_eq _ _ x d (proj1 Hen) Hd).
                 reflexivity.
              ** exact Hinner.
           ++ (* No local allocation follows. *)
              rewrite (ru_let_set fx F' sol' cc cds b ucids codes ub next en'
                         x kind d (erase [] body)
                         (pending_plain_rel en en' x d Hen Hpe) Hset
                         (eq_trans (projections_of_set_erase x d body Hsetp)
                            Hprojs)
                         (not_unboxed_name ub x Hx)).
              rewrite (dead_var_eq sol sol' Hsrc), Hdead,
                (var_used_eq sol sol' Hu).
              assert (IHb : erase [] (RU (ubind_variables en [(x, kind)])
                                        body)
                            = RU' (ubind_variables en' [(x, kind)])
                                (erase [] body))
                by (apply IH; [ exact Hla | exact Hsb | exact Hbk'
                              | exact Hsk' | apply uenv_rel_bind, Hen
                              | apply uenv_plain_bind ]).
              destruct (is_var_used sol x kind); [ | exact IHb ].
              rewrite (erase_let_plain _ _ _ _ _).
              2: { rewrite rebuild_set_u_nil. apply rebuild_set_plain, Hrp. }
              rewrite IHb, (rebuild_set_u_eq _ _ x d (proj1 Hen) Hd).
              reflexivity.
      * (* Not a set. *)
        rewrite (set_binders_let_other _ _ _ _ Hset) in Hsk.
        rewrite (ru_let_other fx F sol cc cds b ucids codes ub next en x kind
                   d body (pending_plain en x d Hpe) Hset).
        rewrite (ru_let_other fx F' sol' cc cds b ucids codes ub next en' x
                   kind d (erase [] body) (pending_plain_rel en en' x d Hen Hpe)
                   Hset).
        rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
        destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
        unfold la_check in Hla.
        destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
        -- (* A local allocation follows. *)
           apply la_body_some in Hlab. subst body.
           apply andb_true_iff in Hla as [Hla Hb'].
           apply andb_true_iff in Hla as [Hla Hargs].
           apply andb_true_iff in Hla as [Hla Hk'].
           apply andb_true_iff in Hla as [_ Hy].
           apply Pos.eqb_eq in Hy. subst y.
           apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
           destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
           apply Pos.eqb_eq in Hargs. subst x'.
           cbn [sets_alone projections_of_set] in Hsb.
           rewrite andb_true_l in Hsb.
           assert (Hbk'' : begins_kept ub (region_binders body'))
             by (intros r Hr; apply Hbk'; exact Hr).
           assert (Hsk'' : begins_kept ub (set_binders body'))
             by (intros r Hr; apply Hsk; exact Hr).
           cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
           set (en1 := ubind_variables en [(x, kind)]).
           assert (IHb : erase [] (RU (ubind_variables en1 [(x, kind)]) body')
                         = RU' (ubind_variables en' [(x, kind)])
                             (erase [] body')).
           { apply IH2 with (y := x) (kind' := kind)
               (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
               [ reflexivity | exact Hb' | exact Hsb | exact Hbk''
               | exact Hsk'' | apply uenv_rel_twice, Hen
               | apply uenv_plain_bind ]. }
           rewrite (ru_let_other fx F sol cc cds b ucids codes ub next en1 x
                      kind
                      (Prim (Region_primitive Local_allocation) (Var x :: rs))
                      body' (pending_plain en1 x _ (uenv_plain_bind _ _))
                      eq_refl).
           rewrite Hdead.
           destruct (is_var_used sol x kind) eqn:Hused; [ | exact IHb ].
           destruct (unboxed_of_node ub (Node_variable x)) as [tb | ] eqn:Hub.
           ++ cbn [rebuild_unboxed_binding].
              rewrite (rebuild_unboxed_binding_erase _ _ _ _ _ _ Hrp), IHb.
              symmetry.
              apply (rebuild_unboxed_binding_env _ _ _ (proj1 Hen) _ _ _ Hd).
           ++ rewrite (rebuild_let_u_erase _ _ _ _ _ _ _ _ Hrp).
              assert (Hinner :
                erase []
                  (rebuild_let_u sol ub [] (ue_env en1) x kind
                     (Prim (Region_primitive Local_allocation) (Var x :: rs))
                     (RU (ubind_variables en1 [(x, kind)]) body'))
                = erase [] (RU (ubind_variables en1 [(x, kind)]) body')).
              { unfold rebuild_let_u. cbn [field_read rebuild_named map].
                change (ue_env en1)
                  with (bind_variables (ue_env en) [(x, kind)]).
                rewrite (rewrite_bound_used sol ub (ue_env en) x kind Hused
                           Hub).
                cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
                reflexivity. }
              rewrite Hinner, IHb.
              symmetry.
              apply (rebuild_let_u_env sol sol' ub Hu Hfu _ _ (proj1 Hen) _ _
                       _ _ Hd).
        -- (* No local allocation follows. *)
           assert (IHb : erase [] (RU (ubind_variables en [(x, kind)]) body)
                         = RU' (ubind_variables en' [(x, kind)])
                             (erase [] body))
             by (apply IH; [ exact Hla | exact Hsb | exact Hbk' | exact Hsk
                           | apply uenv_rel_bind, Hen
                           | apply uenv_plain_bind ]).
           destruct (is_var_used sol x kind); [ | exact IHb ].
           destruct (unboxed_of_node ub (Node_variable x)) as [tb | ].
           ++ rewrite (rebuild_unboxed_binding_erase _ _ _ _ _ _ Hrp), IHb.
              symmetry.
              apply (rebuild_unboxed_binding_env _ _ _ (proj1 Hen) _ _ _ Hd).
           ++ rewrite (rebuild_let_u_erase _ _ _ _ _ _ _ _ Hrp), IHb.
              symmetry.
              apply (rebuild_let_u_env sol sol' ub Hu Hfu _ _ (proj1 Hen) _ _
                       _ _ Hd).
  - (* Let_static *)
    intros bindings body Hcodes IH en en' Hpl Hsa Hbk Hsk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    cbn [sets_alone] in Hsa. apply andb_true_iff in Hsa as [Hbsa Hsb].
    cbn [region_binders] in Hbk. cbn [set_binders] in Hsk.
    rewrite ru_let_static. cbn [erase]. rewrite ru_let_static. f_equal.
    + assert (Hbk1 : begins_kept ub (flat_map binding_region_binders bindings))
        by (intros r Hr; apply Hbk, in_or_app; left; exact Hr).
      assert (Hsk1 : begins_kept ub (flat_map binding_set_binders bindings))
        by (intros r Hr; apply Hsk, in_or_app; left; exact Hr).
      clear IH Hb Hsb Hpe Hbk Hsk.
      induction bindings as [ | bd bds IHb]; [ reflexivity | ].
      cbn [forallb] in Hbs, Hbsa. apply andb_true_iff in Hbs as [Hbd Hbs].
      apply andb_true_iff in Hbsa as [Hbda Hbsa].
      cbn [flat_map] in Hbk1, Hsk1.
      cbn [flat_map map]. rewrite map_app. f_equal.
      * destruct bd as [id c | id | sym const].
        -- destruct c as [params mc ret exn md cbody].
           cbn [erase_binding erase_code].
           rewrite !rb_code, (code_id_used_eq sol sol' Hu).
           destruct (is_code_id_used sol id); cbn [map]; [ | reflexivity ].
           cbn [erase_binding]. do 2 f_equal.
           cbn [binding_regions_plain code_regions_plain] in Hbd.
           apply andb_true_iff in Hbd as [_ Hcb].
           cbn [binding_sets_alone code_sets_alone] in Hbda.
           cbn [binding_region_binders code_region_binders] in Hbk1.
           cbn [binding_set_binders code_set_binders] in Hsk1.
           cbn [rebuild_code_u].
           rewrite !(code_change_u_eq sol sol' cc cds ucids ub Hu Hcds).
           rewrite !(code_continuation_decisions_eq sol sol' cds ub Hu Hcds).
           rewrite (existsb_ext_in
                      (fun '(x, kind) => is_dead_var sol' x kind)
                      (fun '(x, kind) => is_dead_var sol x kind))
             by (intros [x k] _; apply (dead_var_eq sol sol' Hsrc)).
           destruct (code_change_u sol cc cds ucids ub id md) as [ | pd rd];
             cbn [erase_code]; f_equal;
             (match goal with
              | |- context [ if existsb ?f ?l then _ else _ ] =>
                destruct (existsb f l)
              end; [ reflexivity | ]);
             (apply (Hcodes id (Code params mc ret exn md cbody)
                       (or_introl eq_refl));
              [ | exact Hbda
              | intros r Hr; apply Hbk1, in_or_app; left; exact Hr
              | intros r Hr; apply Hsk1, in_or_app; left; exact Hr
              | apply uenv_rel_refl | apply uenv_plain_bind ]);
             cbn [code_body ue_env ubind_variables ubind_variables_then
                    ubind_set bind_variables_then env_kinds];
             (match goal with
              | |- regions_plain (rev ?l ++ env_kinds (ue_env ?EN)) _ = true =>
                assert (HK : env_kinds (ue_env EN) = [])
                  by (case fx; reflexivity);
                rewrite HK, app_nil_r
              end);
             exact Hcb.
        -- reflexivity.
        -- cbn [erase_binding].
           rewrite !rb_symbol, (static_const_is_used_eq sol sol' Hu).
           destruct (static_const_is_used sol sym const); [ | reflexivity ].
           cbn [map erase_binding]. cbn [binding_regions_plain] in Hbd.
           rewrite (rebuild_static_const_eq _ _ (hu_u sol sol' ub Hu) Hfu _ _
                      (proj1 Hen) _ _ Hbd).
           reflexivity.
      * apply IHb; [ | exact Hbs | exact Hbsa
                   | intros r Hr; apply Hbk1, in_or_app; right; exact Hr
                   | intros r Hr; apply Hsk1, in_or_app; right; exact Hr ].
        intros id c Hin. apply (Hcodes id c). right. exact Hin.
    + apply IH; [ exact Hb | exact Hsb
                | intros r Hr; apply Hbk, in_or_app; right; exact Hr
                | intros r Hr; apply Hsk, in_or_app; right; exact Hr
                | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en en' Hpl Hsa Hbk Hsk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    cbn [sets_alone] in Hsa. apply andb_true_iff in Hsa as [Hha Hsb].
    cbn [region_binders] in Hbk. cbn [set_binders] in Hsk.
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    rewrite !ru_let_cont.
    rewrite (handler_exn_erase h [] Hhk),
      (handler_decisions_erase _ _ _ _ h [] Hhk),
      (handler_decisions_eq sol sol' ub Hu).
    set (ds := handler_decisions fx sol ub (handler_is_exn_handler h) h).
    assert (Hds : forallb decision_plain ds = true)
      by exact (handler_decisions_plain ub Hleaves fx sol _ h Hhk).
    assert (IHb : erase [] (RU (ubind_continuations en [(k, ds)]) body)
                  = RU' (ubind_continuations en' [(k, ds)]) (erase [] body))
      by (apply IH; [ exact Hb | exact Hsb
                    | intros r Hr; apply Hbk, in_or_app; right; exact Hr
                    | intros r Hr; apply Hsk, in_or_app; right; exact Hr
                    | apply uenv_rel_conts, Hen | apply uenv_plain_conts ]).
    rewrite <- IHb, (proj1 free_continuations_erase).
    destruct (among k (free_continuations_of_expr
                         (RU (ubind_continuations en [(k, ds)]) body)));
      [ | reflexivity ].
    cbn [erase]. f_equal.
    + apply IHh; [ exact Hh | exact Hha
                 | intros r Hr; apply Hbk, in_or_app; left; exact Hr
                 | intros r Hr; apply Hsk, in_or_app; left; exact Hr
                 | exact Hds | exact Hen ].
    + apply erase_plain, conts_plain_cons; [ | apply conts_plain_nil ].
      apply rebuilt_u_handler_places; assumption.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en en' Hpl Hsa Hbk Hsk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    cbn [sets_alone] in Hsa. apply andb_true_iff in Hsa as [Hhsa Hsb].
    cbn [region_binders] in Hbk. cbn [set_binders] in Hsk.
    rewrite forallb_forall in Hhs, Hhsa.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    rewrite ru_let_cont_rec. cbn [erase]. rewrite ru_let_cont_rec.
    assert (Hds_eq :
      map (fun '(k, h) => (k, handler_decisions fx sol' ub false h))
        (map (fun '(k, h) =>
                (k, erase_handler
                      (map (fun '(k0, h0) => (k0, region_params h0)) handlers
                       ++ []) h)) handlers)
      = map (fun '(k, h) => (k, handler_decisions fx sol ub false h))
          handlers).
    { rewrite map_map. apply map_ext_in. intros [k h] Hin. cbn beta iota.
      rewrite (handler_decisions_erase _ _ _ _ h _ (Hplain k h Hin)),
        (handler_decisions_eq sol sol' ub Hu). reflexivity. }
    rewrite Hds_eq.
    set (en1 := ubind_continuations en
                  (map (fun '(k, h) =>
                          (k, handler_decisions fx sol ub false h))
                     handlers)).
    set (en1' := ubind_continuations en'
                   (map (fun '(k, h) =>
                           (k, handler_decisions fx sol ub false h))
                      handlers)).
    assert (Hen1 : uenv_rel en1 en1') by (apply uenv_rel_conts, Hen).
    f_equal.
    + rewrite !map_map. apply map_ext_in.
      intros [k h] Hin. cbn beta iota.
      assert (Hds : forallb decision_plain
                      (handler_decisions fx sol ub false h) = true)
        by exact (handler_decisions_plain ub Hleaves fx sol _ h
                    (Hplain k h Hin)).
      rewrite (erase_handler_plain _ h)
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply region_places_plain; exact (Hplain k0 h0 Hin0)).
      rewrite (erase_handler_plain _ (rebuild_handler_u _ _ _ _ _ _ _ _ _ _ _
                                        _ _ _ _ _))
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply rebuilt_u_handler_places;
            [ exact (Hplain k0 h0 Hin0)
            | exact (handler_decisions_plain ub Hleaves fx sol _ h0
                       (Hplain k0 h0 Hin0)) ]).
      rewrite (handler_decisions_erase _ _ _ _ h [] (Hplain k h Hin)),
        (handler_decisions_eq sol sol' ub Hu).
      f_equal. apply (IHh k h Hin); [ exact (Hhs (k, h) Hin)
                                     | exact (Hhsa (k, h) Hin)
                                     | | | exact Hds | exact Hen1 ].
      * intros r Hr. apply Hbk. apply in_or_app. left.
        apply in_flat_map. exists (k, h). split; [ exact Hin | exact Hr ].
      * intros r Hr. apply Hsk. apply in_or_app. left.
        apply in_flat_map. exists (k, h). split; [ exact Hin | exact Hr ].
    + rewrite (erase_plain _ body).
      2: { apply conts_plain_app; [ | apply conts_plain_nil ].
           intros k p Hin. apply in_map_iff in Hin as [[k' h] [Heq Hin]].
           injection Heq as <- <-. apply region_places_plain.
           exact (Hplain k' h Hin). }
      rewrite (erase_plain _ (RU en1 body)).
      2: { apply conts_plain_app; [ | apply conts_plain_nil ].
           intros k p Hin. apply in_map_iff in Hin as [[k' h'] [Heq Hin]].
           injection Heq as <- <-.
           apply in_map_iff in Hin as [[k0 h0] [Heq Hin]].
           injection Heq as <- <-. apply rebuilt_u_handler_places;
             [ exact (Hplain k0 h0 Hin)
             | exact (handler_decisions_plain ub Hleaves fx sol _ h0
                        (Hplain k0 h0 Hin)) ]. }
      apply IH; [ exact Hb | exact Hsb
                | intros r Hr; apply Hbk, in_or_app; right; exact Hr
                | intros r Hr; apply Hsk, in_or_app; right; exact Hr
                | exact Hen1 | apply uenv_plain_conts ].
  - (* Apply *)
    intros ap en en' Hpl _ _ _ Hen _.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Hr].
    cbn [rebuild_expr_u erase].
    rewrite erase_rebuild_apply_u by assumption.
    symmetry.
    eapply rebuild_apply_u_eq; eassumption.
  - (* Apply_cont *)
    intros ac en en' Hpl _ _ _ Hen Hpe. cbn [regions_plain] in Hpl.
    cbn [rebuild_expr_u erase]. rewrite erase_rebuild_apply_cont_u,
      erase_apply_cont_nil.
    symmetry. exact (rebuild_apply_cont_u_eq sol sol' ub Hu Hsrc _ _ ac Hen
                       Hpl).
  - (* Switch *)
    intros scrutinee arms en en' Hpl _ _ _ Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Harms].
    cbn [rebuild_expr_u erase]. rewrite erase_rebuild_switch_u, map_arms_id.
    symmetry.
    exact (rebuild_switch_u_eq sol sol' ub Hu Hsrc _ _ scrutinee arms Hen Hs
             Harms).
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en en' dc ds Hpl Hsa Hbk Hsk Hds Hen.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    rewrite !rh_unfold.
    rewrite erase_handler_no_region.
    2: { apply with_params_kinds_plain.
         apply params_of_decisions_plain; [ | exact Hds ].
         apply with_kinds_plain, Hk. }
    f_equal.
    rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
               (fun '(x, kind) => is_dead_var sol x kind))
      by (intros [x k] _; apply (dead_var_eq sol sol' Hsrc)).
    destruct (dc && _); [ reflexivity | ].
    apply IH; [ exact Hb | exact Hsa | exact Hbk | exact Hsk
              | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
Qed.

End CallsCommute.
