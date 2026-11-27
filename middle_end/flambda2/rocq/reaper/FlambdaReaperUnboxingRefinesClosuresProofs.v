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

(* The proofs of FlambdaReaperUnboxingRefinesClosuresSpec.v, and the
   construction of the relation for the wider fragments of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v and
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v.

   The first statement relates the two rebuilds of the unit, the change
   of calling conventions alone ([cc_rebuild]) and the rebuild with
   unboxing and changes of representation in the mode of the fix or of
   its variant ([u_rebuild]), by the relation of
   FlambdaReaperUnboxingRefinesClosuresSimProofs.v ([rr]), a simulation.
   [rebuild_rr] builds it by induction on a unit of the wider fragment
   of FlambdaReaperUnboxingRefinesClosuresReachSpec.v ([repr_expr_r], for
   the code [reach] that the flow may bring to the callee of a call whose
   code is not known, [RCHf]), the continuations in scope being described
   by their
   decisions ([cinv]): those that may be applied have, in the second
   rebuild, decisions that read as kept a parameter kept with no use
   ([normed]), and in the first the masks of those decisions, or none for
   the return continuation of code that keeps its calling convention,
   whose decisions keep every result; the exception continuation of code
   is described in the second rebuild when it may be raised to. The
   unboxed names are those of the unboxing, with the changed names added
   with no field ([M']); a name is changed ([CHf]) if the changes of
   representation give it one and it is not the variable of an unboxed
   field. The two rebuilt terms agree but where the second unboxes, as in
   FlambdaReaperUnboxingRefinesProofs.v, at a set whose representation
   changes, whose new slots are those of [new_slots] ([changed_slots]),
   and at the reads of its slots. Calls and continuations are the same in
   both rebuilds: the arguments kept by the decisions of the callee
   ([get_args_with_kinds_c]) or, for code that keeps its calling
   convention, at the same mask in both ([kept_args_ok]); the extra
   arguments of the exception continuation that its handler keeps
   ([extra_rebuilt]); and no wrapper in either ([wrapper_cc_none],
   [wrapper_u_none], [make_apply_wrapper_none]); but a call of code that
   never returns, which the second makes return to a handler [Invalid]
   ([rr_call_never_r], [rr_call_never_wraps]). A call whose code is not
   known is rebuilt alike by both, at the mask read off its callee
   ([ind_call_rebuilt]). Code that keeps its
   calling convention has the same parameters and metadata in both
   ([code_rebuilt]). The relation is built at continuations [D] that are
   not live in the original term ([rebuild_ok], [dead_in], [live_conts]):
   the return continuation of code taken to never return ([nr],
   [code_dead]), which the fragment checks. A handler that only the first
   rebuild keeps is dead ([rr_let_cont_never], [rr_dead_shadow]); one that
   only the second keeps cannot be ([rr_shape]). No continuation of the
   unit is a wrapper's ([conts_ok]). The fragment of
   FlambdaReaperUnboxingRefinesClosuresSpec.v is within that of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v, and that within the
   wider one, for any flow (section [Within], [repr_within_unit],
   [repr_calls_within_reach_unit]), so that their statements follow
   ([rebuild_repr_modes_with_refines_cc] from
   [rebuild_repr_calls_modes_with_refines_cc], and that from
   [rebuild_repr_reach_modes_with_refines_cc] where the flow may bring any
   code anywhere). The second statement
   composes the first with
   FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably,
   in its version with local fields ([_local]), of which the statement
   with the old premise is a corollary; the statements with no code
   taken to never return are those with [nr] empty. The examples are the
   statements applied, or runs.
   The two rebuilds rewrite what the parameters that they keep state as
   [types] says, the same for both (the section [Rebuilds]): the handlers
   and the code of the fragment have decided decisions, read as the
   rebuilds read them, or keep their calling convention, so that they
   keep the same parameters in the two rebuilds and state the same of
   them ([u_handler_metadata_decided], [decisions_subkinds_decided],
   [kept_code_metadata]). [rebuild_repr_modes_with_refines_cc] holds for
   any [types], [rebuild_repr_modes_refines_cc] is its instance without
   types. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaRenameApartSpec FlambdaKindModel
  FlambdaModel FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesClosuresReachSpec
  FlambdaReaperUnboxingRefinesClosuresSimProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaNameAbstractionSpec
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperTupledProofs.
Import ListNotations.

(* ======================================================================== *)
(* Equalities                                                               *)
(* ======================================================================== *)

Lemma unboxed_field_eqb_eq (u : unboxed_field) :
  ∀ u' : unboxed_field, unboxed_field_eqb u u' = true -> u = u'.
Proof.
  induction u as [x | fields IH] using unboxed_field_rec;
    intros [x' | fields'] H; cbn [unboxed_field_eqb] in H;
    try discriminate H.
  - apply Pos.eqb_eq in H. subst x'. reflexivity.
  - f_equal. revert fields' H.
    induction fields as [ | [f u] fields IHf]; intros [ | [f' u'] fields'] H;
      try discriminate H; [ reflexivity | ].
    apply andb_true_iff in H as [H Hrest].
    apply andb_true_iff in H as [Hf Hu].
    apply field_eqb_eq in Hf. subst f'.
    rewrite (IH f u (or_introl eq_refl) u' Hu).
    rewrite (IHf (fun f0 u0 Hin => IH f0 u0 (or_intror Hin)) fields' Hrest).
    reflexivity.
Qed.

Lemma slot_pairs_eqb_eq (l : list (function_slot * function_slot)) :
  ∀ l', slot_pairs_eqb l l' = true -> l = l'.
Proof.
  induction l as [ | [a b] l IH]; intros [ | [a' b'] l'] H;
    try discriminate H; [ reflexivity | ].
  cbn in H. apply andb_true_iff in H as [H Hl].
  apply andb_true_iff in H as [Ha Hb].
  apply Pos.eqb_eq in Ha, Hb. subst. rewrite (IH l' Hl). reflexivity.
Qed.

Lemma repr_option_eqb_eq (o o' : option closure_repr) :
  repr_option_eqb o o' = true -> o = o'.
Proof.
  destruct o as [[fs fss cur] | ], o' as [[fs' fss' cur'] | ]; cbn;
    intros H; try discriminate H; [ | reflexivity ].
  unfold closure_repr_eqb in H.
  cbn [cr_fields cr_function_slots cr_current] in H.
  apply andb_true_iff in H as [H Hc].
  apply andb_true_iff in H as [Hf Hs].
  pose proof (unboxed_field_eqb_eq (Unboxed fs) (Unboxed fs') Hf) as Hf'.
  injection Hf' as ->.
  apply slot_pairs_eqb_eq in Hs. subst fss'.
  apply Pos.eqb_eq in Hc. subst cur'. reflexivity.
Qed.

Lemma option_code_eqb_eq (o o' : option code_id) :
  option_code_eqb o o' = true -> o = o'.
Proof.
  destruct o, o'; cbn; intros H; try discriminate H; [ | reflexivity ].
  apply Pos.eqb_eq in H. subst. reflexivity.
Qed.

Lemma forallb2_length {A B : Type} (p : A -> B -> bool) (l : list A) :
  ∀ l' : list B, forallb2 p l l' = true -> length l = length l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] H; cbn in H |- *;
    try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [_ H]. rewrite (IH l' H). reflexivity.
Qed.

Lemma flat_map_ext_in' {A B : Type} (f g : A -> list B) (l : list A) :
  (∀ a : A, a ∈ l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | a l IH]; intros H; [ reflexivity | ].
  cbn [flat_map]. rewrite (H a (or_introl eq_refl)), IH;
    [ reflexivity | intros b Hb; exact (H b (or_intror Hb)) ].
Qed.

Lemma group_by_sizes_map' {A B : Type} (f : A -> B) (sizes : list nat) :
  ∀ l : list A,
    group_by_sizes sizes (map f l) = map (map f) (group_by_sizes sizes l).
Proof.
  induction sizes as [ | n sizes IH]; intros l; [ reflexivity | ].
  cbn [group_by_sizes map]. rewrite firstn_map, skipn_map, IH. reflexivity.
Qed.

Lemma group_by_parameter_map' {A B : Type} (f : A -> B) (a : arity)
    (l : list A) :
  group_by_parameter a (map f l) = map (map f) (group_by_parameter a l).
Proof. apply group_by_sizes_map'. Qed.

Lemma prepend_nil' {A : Type} (groups : list (list A)) :
  prepend_first [] groups = groups.
Proof. destruct groups; reflexivity. Qed.

Lemma decisions_kinds_decided (ds : list param_decision) :
  decisions_kinds (map Decided ds) = kept_kinds ds.
Proof.
  induction ds as [ | [k | ] ds IH]; [ reflexivity | | ];
    cbn [map decisions_kinds flat_map]; unfold decisions_kinds in IH;
    rewrite IH; reflexivity.
Qed.

Lemma arity_of_decisions_decided (ds : list param_decision) :
  arity_of_decisions_u (map Decided ds) = arity_of_decisions ds.
Proof.
  unfold arity_of_decisions_u, arity_of_decisions.
  rewrite decisions_kinds_decided. reflexivity.
Qed.

Lemma arity_of_grouped_decided (gs : list (list param_decision)) :
  arity_of_grouped_decisions_u
    (prepend_first [Decided Delete] (map (map Decided) gs))
  = arity_of_grouped_decisions gs.
Proof.
  unfold arity_of_grouped_decisions_u, arity_of_grouped_decisions.
  destruct gs as [ | g gs]; [ reflexivity | ].
  cbn [map prepend_first app]. f_equal.
  - change (decisions_kinds (Decided Delete :: map Decided g))
      with (decisions_kinds (map Decided g)).
    rewrite decisions_kinds_decided. reflexivity.
  - rewrite map_map. apply map_ext. intros g'.
    rewrite decisions_kinds_decided. reflexivity.
Qed.

Lemma group_by_sizes_length' {A : Type} (sizes : list nat) :
  ∀ l : list A, length (group_by_sizes sizes l) = length sizes.
Proof.
  induction sizes as [ | n sizes IH]; intros l; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma group_by_parameter_length' {A : Type} (a : arity) (l : list A) :
  length (group_by_parameter a l) = length a.
Proof.
  unfold group_by_parameter. rewrite group_by_sizes_length'.
  unfold unarize_per_parameter. rewrite !length_map. reflexivity.
Qed.

(* ======================================================================== *)
(* The names of the relation                                                *)
(* ======================================================================== *)

Definition empty_fields (changed : list (node * closure_repr))
    : list (node * unboxed_fields) :=
  map (fun '(n, _) => (n, [])) changed.

Section Names.

Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).

(* The unboxed names, and the changed names with no field. *)
Definition M' : list (node * unboxed_fields) :=
  empty_fields changed ++ unboxed.

(* The changed variables, but for those of unboxed fields. *)
Definition CHf (x : variable) : option closure_repr :=
  if among x (field_variables unboxed) then None
  else changed_of_node changed (Node_variable x).

Lemma lookup_empty_fields (n : node) :
  Keyed.lookup node_eqb n (empty_fields changed)
  = option_map (fun _ => []) (Keyed.lookup node_eqb n changed).
Proof.
  unfold empty_fields. induction changed as [ | [n0 r] l IH];
    [ reflexivity | ].
  cbn [map]. rewrite !keyed_lookup_cons.
  destruct (node_eqb n n0); [ reflexivity | exact IH ].
Qed.

Lemma lookup_M' (n : node) :
  Keyed.lookup node_eqb n M'
  = match Keyed.lookup node_eqb n changed with
    | Some _ => Some []
    | None => Keyed.lookup node_eqb n unboxed
    end.
Proof.
  unfold M'. rewrite keyed_lookup_app, lookup_empty_fields.
  destruct (Keyed.lookup node_eqb n changed); reflexivity.
Qed.

Lemma field_variables_M' : field_variables M' = field_variables unboxed.
Proof.
  unfold M', field_variables. rewrite flat_map_app.
  replace (flat_map (fun '(_, fields) => map fst (fields_leaves fields))
             (empty_fields changed))
    with (@nil variable); [ reflexivity | ].
  unfold empty_fields. induction changed as [ | [n r] l IH];
    [ reflexivity | exact IH ].
Qed.

Lemma fields_of_M' (x : variable) :
  fields_of M' x
  = match changed_of_node changed (Node_variable x) with
    | Some _ => Some []
    | None => fields_of unboxed x
    end.
Proof. unfold fields_of. rewrite lookup_M'. reflexivity. Qed.

Lemma CHf_M (x : variable) (r : closure_repr) :
  CHf x = Some r -> fields_of M' x = Some [].
Proof.
  unfold CHf. destruct (among x (field_variables unboxed)); intros H;
    [ discriminate H | ].
  rewrite fields_of_M'. unfold changed_of_node in H |- *. rewrite H.
  reflexivity.
Qed.

Lemma CHf_fresh (x : variable) (r : closure_repr) :
  CHf x = Some r -> among x (field_variables M') = false.
Proof.
  unfold CHf. rewrite field_variables_M'.
  destruct (among x (field_variables unboxed)); intros H;
    [ discriminate H | reflexivity ].
Qed.

Lemma CHf_var (x : variable) :
  among x (field_variables unboxed) = false ->
  CHf x = changed_of_node changed (Node_variable x).
Proof. unfold CHf. intros ->. reflexivity. Qed.

Lemma plain_variable_M' (x : variable) :
  plain_variable M' x
  = plain_variable unboxed x
    && is_none (changed_of_node changed (Node_variable x)).
Proof.
  unfold plain_variable. rewrite field_variables_M'.
  unfold Keyed.is_bound. rewrite lookup_M'.
  unfold changed_of_node.
  destruct (Keyed.lookup node_eqb (Node_variable x) changed);
    cbn [is_none]; [ | rewrite andb_true_r; reflexivity ].
  rewrite andb_false_r. reflexivity.
Qed.

(* A simple of the fragment, whose variable is not that of a field, is
   plain in the relation. *)
Lemma plain_c_M' (s : simple) :
  plain_c unboxed changed s = true ->
  (∀ z : variable, z ∈ variables_in_simple s ->
     among z (field_variables unboxed) = false) ->
  plain_simple M' s = true.
Proof.
  unfold plain_c. intros H Hz. apply andb_true_iff in H as [Hp Hc].
  pose proof (strong_simple unboxed s Hp Hz) as Hs.
  destruct s as [x | sym | c]; cbn [plain_simple not_changed] in *;
    [ | | reflexivity ].
  - rewrite plain_variable_M', Hs. exact Hc.
  - unfold Keyed.is_bound in *. rewrite lookup_M'.
    unfold changed_name, changed_of_node in Hc.
    destruct (Keyed.lookup node_eqb (Node_symbol sym) changed);
      [ discriminate Hc | exact Hs ].
Qed.

Lemma plain_c_var_M' (x : variable) :
  plain_c unboxed changed (Var x) = true ->
  among x (field_variables unboxed) = false ->
  plain_variable M' x = true.
Proof.
  intros H Hx. apply (plain_c_M' (Var x) H).
  intros z [<- | []]. exact Hx.
Qed.

End Names.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment                               *)
(* ======================================================================== *)

Section Rebuilds.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable pdc : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable ff ffu : positive.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable kept_slots : value_slot -> bool.
Variable codes0 : list (code_id * code).
Variable symbols : list symbol.
Variable is_local : field -> bool.
Variable nr : code_id -> bool.
(* How the two rebuilds rewrite what the parameters that they keep state,
   the same for both (FlambdaReaperRebuild.subkinds_rewriting). *)
Variable types : subkinds_rewriting.
(* The code that the flow may bring to the callee of a call whose code is
   not known (FlambdaReaperUnboxingRefinesClosuresReachSpec.v). *)
Variable reach : bool -> variable -> code_id -> bool.
Hypothesis sol_closed : closed is_local (graph_of_facts F) sol.
Hypothesis local_slots : is_local_slots_only is_local.
Hypothesis fields_read : ∀ (x : variable) (fields : unboxed_fields)
    (f : field) (u : unboxed_field),
  fields_of unboxed x = Some fields -> (f, u) ∈ fields ->
  ∃ w base : node,
    g_accessor (graph_of_facts F) w f base
    ∧ s_usages sol (Node_variable x) base = true
    ∧ s_has_usage sol w = true.
Hypothesis reads_found : ∀ (x y : variable) (kind : flambda_kind)
    (f : field) (fields : unboxed_fields),
  fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
  g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
  is_block_like_field f = true ->
  is_dead_var sol x kind = false -> is_var_used sol x kind = true ->
  kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
  ∃ v : variable, Keyed.lookup field_eqb f fields = Some (Not_unboxed v).
Hypothesis my_closures_fresh : ∀ (id : code_id) (mc : variable),
  lookup id my_closures = Some mc -> among mc (field_variables unboxed) = false.

Abbreviation edges ctx ce e :=
  (FlambdaReaperFlowProofs.expr_edges (graph_of_facts F) noted deps cds
     kept_slots codes0 symbols True False ctx ce e).
Abbreviation sol_u := (solution_without_unboxed sol unboxed).
Abbreviation fresh z := (among z (field_variables unboxed) = false).
Abbreviation MM := (M' unboxed changed).
Abbreviation CH := (CHf unboxed changed).
Abbreviation repr kinds C e :=
  (repr_expr_r F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr reach kinds C e).
Abbreviation live e := (live_conts F sol code_deps codes e).

(* The code of the unit, as calls see it. *)
Definition CSf (id : code_id) : option code_info :=
  match lookup id codes, lookup id my_closures with
  | Some md, Some mc =>
    Some (Code_info (CH mc) (length (cm_params_arity md)) (nr id))
  | _, _ => None
  end.

(* The code that the flow may bring to the callee of a call. *)
Definition RCHf (b : bool) (f : variable) (id : code_id) : Prop :=
  reach b f id = true.

Definition cc_rebuild (en : rebuild_env) (e : expr) : expr :=
  rebuild_expr_cc F sol cannot_change code_deps pdc unit_code_ids codes ff types
    en
    e.

Definition u_rebuild (en : unbox_env) (e : expr) : expr :=
  rebuild_expr_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu changed types en e.

(* The continuations in scope ([C]): those that may be applied have, in
   the second rebuild, decisions that [C] has once read as the rebuilds
   read them ([normed]), and in the first their masks, or none if they
   keep every parameter (the return continuation of code that keeps its
   calling convention); so has the exception continuation of code that
   may be raised to; the exception continuation of code that may not has
   decisions in the second only; the others have none in either. *)
Record cinv (C : finite_map (list decision * bool)) (en : rebuild_env)
    (enu : unbox_env) : Prop := {
  ci_kinds : env_kinds (ue_env enu) = env_kinds en;
  ci_proj : env_projections en = [];
  ci_proj_u : env_projections (ue_env enu) = [];
  ci_drop : ue_drop_projections enu = false;
  ci_app : ∀ (k : continuation) (ds : list decision) (b : bool),
    lookup k C = Some (ds, b) -> b = true ∨ ds <> [] ->
    (∃ dsu : list decision,
       lookup k (ue_decisions enu) = Some dsu ∧ normed dsu = ds)
    ∧ (lookup k (env_conts en) = Some (map is_kept_decision ds)
       ∨ lookup k (env_conts en) = None ∧ forallb keeps ds = true)
    ∧ forallb decided ds = true;
  ci_out : ∀ k : continuation,
    lookup k C = None ->
    lookup k (ue_decisions enu) = None ∧ lookup k (env_conts en) = None
}.

Definition names_ok (e : expr) : Prop :=
  ∀ z : variable, z ∈ variables_of e -> fresh z.

Definition names_ok_h (h : continuation_handler) : Prop :=
  ∀ z : variable, z ∈ variables_in_handler h -> fresh z.

(* No continuation of the term is the continuation of a wrapper. *)
Definition conts_ok (l : list continuation) : Prop :=
  ∀ k : continuation, k ∈ l -> k <> ff ∧ k <> ffu.

(* The continuations [D] are not live in the term. *)
Definition dead_in (D l : list continuation) : Prop :=
  ∀ k : continuation, k ∈ D -> k ∉ l.

Lemma conts_ok_incl (l l' : list continuation) :
  incl l l' -> conts_ok l' -> conts_ok l.
Proof. intros Hi H k Hk. exact (H k (Hi k Hk)). Qed.

Lemma dead_in_incl (D l l' : list continuation) :
  incl l l' -> dead_in D l' -> dead_in D l.
Proof. intros Hi H k Hk Hl. exact (H k Hk (Hi k Hl)). Qed.

Lemma cinv_bind (C : finite_map (list decision * bool)) (en : rebuild_env)
    (enu : unbox_env) (bound : list (variable * flambda_kind)) :
  cinv C en enu ->
  cinv C (bind_variables en bound) (ubind_variables enu bound).
Proof.
  intros Hci. destruct Hci as [Hk Hp Hpu Hd Happ Hout].
  split; try reflexivity.
  - cbn. rewrite Hk. reflexivity.
  - exact Happ.
  - exact Hout.
Qed.

(* ------------------------------------------------------------------------ *)
(* Simples                                                                  *)
(* ------------------------------------------------------------------------ *)

Lemma rewrite_c (en : rebuild_env) (s : simple) :
  plain_c unboxed changed s = true ->
  (∀ z, z ∈ variables_in_simple s -> fresh z) ->
  rewrite_simple sol_u en s = rewrite_simple sol en s.
Proof.
  intros Hs Hz. apply (rewrite_simple_u sol unboxed en s).
  apply andb_true_iff in Hs as [Hs _]. exact (strong_simple unboxed s Hs Hz).
Qed.

Lemma rewrite_plain_M' (en : rebuild_env) (s : simple) :
  plain_simple MM s = true -> plain_simple MM (rewrite_simple sol en s) = true.
Proof.
  destruct s as [x | sym | c]; cbn [rewrite_simple]; intros Hs;
    [ | | exact Hs ].
  - destruct (is_var_used sol x _); [ exact Hs | reflexivity ].
  - destruct (is_symbol_used sol sym); [ exact Hs | reflexivity ].
Qed.

Lemma plain_cs_M' (l : list simple) :
  forallb (plain_c unboxed changed) l = true ->
  (∀ z, z ∈ variables_in_simples l -> fresh z) ->
  forallb (plain_simple MM) l = true.
Proof.
  induction l as [ | s l IH]; intros Hl Hz; [ reflexivity | ].
  cbn [forallb] in Hl |- *. apply andb_true_iff in Hl as [Hs Hl].
  unfold variables_in_simples in Hz. cbn [flat_map] in Hz.
  rewrite (plain_c_M' unboxed changed s Hs
             (fun z H => Hz z (in_or_app _ _ _ (or_introl H)))).
  apply IH; [ exact Hl | ].
  intros z H. apply Hz, in_or_app. right. exact H.
Qed.

Lemma rewrites_plain_M' (en : rebuild_env) (l : list simple) :
  forallb (plain_simple MM) l = true ->
  forallb (plain_simple MM) (map (rewrite_simple sol en) l) = true.
Proof.
  induction l as [ | s l IH]; intros Hl; [ reflexivity | ].
  cbn [forallb map] in Hl |- *. apply andb_true_iff in Hl as [Hs Hl].
  rewrite (rewrite_plain_M' en s Hs), (IH Hl). reflexivity.
Qed.

Lemma plain_cs_strong (l : list simple) :
  forallb (plain_c unboxed changed) l = true ->
  (∀ z, z ∈ variables_in_simples l -> fresh z) ->
  forallb (plain_simple unboxed) l = true.
Proof.
  intros Hl Hz. apply strong_simples; [ | exact Hz ].
  rewrite forallb_forall in Hl |- *. intros s Hin.
  specialize (Hl s Hin). apply andb_true_iff in Hl as [Hl _]. exact Hl.
Qed.

Lemma filter_kept_plain (mask : list bool) (l : list simple) :
  forallb (plain_simple MM) l = true ->
  forallb (plain_simple MM) (filter_kept mask l) = true.
Proof. apply filter_kept_forallb. Qed.


(* ------------------------------------------------------------------------ *)
(* Continuations                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma decided_is_decided (ds : list decision) :
  forallb decided ds = forallb is_decided ds.
Proof. reflexivity. Qed.

(* A parameter kept with no use is given what is given. *)
Lemma get_args_normed (en : rebuild_env) (args : list simple) :
  ∀ ds : list decision,
    get_args sol unboxed en ds args = get_args sol unboxed en (normed ds) args.
Proof.
  induction args as [ | a args IH]; intros [ | d ds]; try reflexivity.
  unfold normed in *. cbn [map get_args]. rewrite (IH ds).
  destruct (get_args sol unboxed en (map norm_decision ds) args);
    [ | reflexivity ].
  destruct d as [[k | ] | | ]; reflexivity.
Qed.

Lemma filter_kept_keeps {A : Type} (ds : list decision) :
  forallb keeps ds = true ->
  ∀ l : list A, filter_kept (map is_kept_decision ds) l = l.
Proof.
  induction ds as [ | d ds IH]; intros Hk l.
  - cbn [map]. induction l as [ | a l IHl]; [ reflexivity | ].
    cbn [filter_kept]. rewrite IHl. reflexivity.
  - destruct l as [ | a l]; [ reflexivity | ].
    cbn [forallb] in Hk. apply andb_true_iff in Hk as [Hd Hk].
    destruct d as [[k | ] | | ]; try discriminate Hd.
    cbn. rewrite (IH Hk l). reflexivity.
Qed.

(* An [Apply_cont] of the fragment is rebuilt alike, with plain
   arguments. *)
Lemma rewrite_apply_cont_c (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ac : apply_cont) :
  cinv C en enu -> apply_cont_ok_w unboxed changed C ac = true ->
  (∀ z, z ∈ variables_in_apply_cont ac -> fresh z) ->
  rewrite_apply_cont_u sol unboxed enu ac = rewrite_apply_cont sol en ac
  ∧ ∀ ac' : apply_cont,
      rewrite_apply_cont sol en ac = Some ac' ->
      forallb (plain_simple MM) (ac_args ac') = true.
Proof.
  intros Hci Hok Hz. unfold apply_cont_ok_w in Hok.
  apply andb_true_iff in Hok as [Hargs Hk].
  assert (Hplain : forallb (plain_simple MM) (ac_args ac) = true)
    by exact (plain_cs_M' (ac_args ac) Hargs Hz).
  assert (Hstrong : forallb (plain_simple unboxed) (ac_args ac) = true)
    by exact (plain_cs_strong (ac_args ac) Hargs Hz).
  assert (Hmap : map (rewrite_simple sol (ue_env enu)) (ac_args ac)
                 = map (rewrite_simple sol en) (ac_args ac))
    by (apply map_ext; intros s;
        apply rewrite_simple_kinds, (ci_kinds _ _ _ Hci)).
  assert (Hmap_u : map (rewrite_simple sol_u (ue_env enu)) (ac_args ac)
                   = map (rewrite_simple sol en) (ac_args ac)).
  { rewrite <- Hmap. apply map_ext_in. intros s Hs.
    rewrite forallb_forall in Hstrong.
    exact (rewrite_simple_u sol unboxed _ s (Hstrong s Hs)). }
  unfold rewrite_apply_cont_u, rewrite_apply_cont.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ split; [ reflexivity | intros ac' H; discriminate H ] | ].
  destruct (lookup (ac_continuation ac) C) as [[ds b] | ] eqn:HC.
  - assert (Hb : b = true ∨ ds <> []).
    { destruct b; [ left; reflexivity | right ].
      destruct ds; [ discriminate Hk | discriminate ]. }
    destruct (ci_app _ _ _ Hci _ _ _ HC Hb) as ((dsu & Hd & Hn) & Hm & Hdec).
    rewrite Hd, get_args_normed, Hn,
      (get_args_decided sol unboxed (ue_env enu) (ac_args ac) Hstrong ds
         Hdec),
      Hmap.
    destruct Hm as [Hm | [Hm Hkeeps]]; rewrite Hm.
    + split; [ reflexivity | ]. intros ac' H. injection H as <-. cbn.
      apply filter_kept_plain, rewrites_plain_M', Hplain.
    + rewrite (filter_kept_keeps ds Hkeeps).
      split; [ reflexivity | ]. intros ac' H. injection H as <-. cbn.
      apply rewrites_plain_M', Hplain.
  - destruct (ci_out _ _ _ Hci _ HC) as (Hd & Hm). rewrite Hd, Hm, Hmap_u.
    split; [ reflexivity | ]. intros ac' H. injection H as <-. cbn.
    apply rewrites_plain_M', Hplain.
Qed.

(* The continuation of an [Apply_cont] that the rebuilds keep is live. *)
Lemma rewrite_apply_cont_live (en : rebuild_env) (ac ac' : apply_cont) :
  rewrite_apply_cont sol en ac = Some ac' ->
  ac_continuation ac' = ac_continuation ac
  ∧ ac_continuation ac ∈ live_apply_cont sol ac.
Proof.
  unfold rewrite_apply_cont, live_apply_cont.
  destruct (existsb (simple_has_no_source sol) (ac_args ac));
    [ intros H; discriminate H | ].
  intros H. injection H as <-. split; [ reflexivity | left; reflexivity ].
Qed.

Lemma apply_cont_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ac : apply_cont)
    (D : list continuation) :
  cinv C en enu -> apply_cont_ok_w unboxed changed C ac = true ->
  (∀ z, z ∈ variables_in_apply_cont ac -> fresh z) ->
  dead_in D (live_apply_cont sol ac) ->
  rr MM CH CSf RCHf D (rebuild_apply_cont sol en ac)
    (rebuild_apply_cont_u sol unboxed enu ac).
Proof.
  intros Hci Hok Hz HD.
  destruct (rewrite_apply_cont_c C en enu ac Hci Hok Hz) as [Heq Hplain].
  unfold rebuild_apply_cont, rebuild_apply_cont_u. rewrite Heq.
  destruct (rewrite_apply_cont sol en ac) as [ac' | ] eqn:Hac;
    [ | constructor ].
  destruct (rewrite_apply_cont_live en ac ac' Hac) as [Hk Hl].
  constructor; [ exact (Hplain ac' eq_refl) | ].
  rewrite Hk. intros Hin. exact (HD _ Hin Hl).
Qed.

Lemma switch_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  cinv C en enu -> plain_c unboxed changed scrutinee = true ->
  forallb (fun '(_, ac) => apply_cont_ok_w unboxed changed C ac) arms = true ->
  (∀ z, z ∈ variables_of (Switch scrutinee arms) -> fresh z) ->
  ∀ D : list continuation,
    dead_in D (live (Switch scrutinee arms)) ->
    rr MM CH CSf RCHf D (rebuild_switch sol en scrutinee arms)
      (rebuild_switch_u sol unboxed enu scrutinee arms).
Proof.
  intros Hci Hs Harms Hz D HD. cbn [variables_of] in Hz.
  assert (Hsz : ∀ z, z ∈ variables_in_simple scrutinee -> fresh z)
    by (intros z H; apply Hz, in_or_app; left; exact H).
  assert (Hin : ∀ d ac, (d, ac) ∈ Keyed.compact Z.eqb arms ->
            apply_cont_ok_w unboxed changed C ac = true
            ∧ (∀ z, z ∈ variables_in_apply_cont ac -> fresh z)
            ∧ dead_in D (live_apply_cont sol ac)).
  { intros d ac H.
    apply (keyed_in_compact Z.eqb
             (fun a b => conj (proj1 (Z.eqb_eq a b))
                            (proj2 (Z.eqb_eq a b))))
      in H.
    apply (keyed_lookup_in Z.eqb
             (fun a b => conj (proj1 (Z.eqb_eq a b))
                            (proj2 (Z.eqb_eq a b))))
      in H.
    rewrite forallb_forall in Harms. split; [ exact (Harms _ H) | ].
    split.
    - intros z Hz'. apply Hz, in_or_app. right. apply in_flat_map.
      exists (d, ac). split; assumption.
    - eapply dead_in_incl; [ | exact HD ]. intros k Hk.
      cbn [live_conts]. apply in_flat_map. exists (d, ac).
      split; assumption. }
  unfold rebuild_switch, rebuild_switch_u.
  set (arms' := Keyed.compact Z.eqb arms) in *.
  assert (Heq : flat_map
                  (fun '(d, ac) =>
                     match rewrite_apply_cont_u sol unboxed enu ac with
                     | Some ac => [(d, ac)]
                     | None => []
                     end) arms'
                = flat_map
                    (fun '(d, ac) =>
                       match rewrite_apply_cont sol en ac with
                       | Some ac => [(d, ac)]
                       | None => []
                       end) arms').
  { apply flat_map_ext_in'. intros [d ac] Hda.
    destruct (Hin d ac Hda) as (Hok & Hz' & _).
    rewrite (proj1 (rewrite_apply_cont_c C en enu ac Hci Hok Hz')).
    reflexivity. }
  rewrite Heq.
  destruct (Keyed.is_empty _); [ constructor | ].
  rewrite (rewrite_c (ue_env enu) scrutinee Hs Hsz).
  rewrite (rewrite_simple_kinds sol (ue_env enu) en scrutinee
             (ci_kinds _ _ _ Hci)).
  constructor.
  - apply rewrite_plain_M'. exact (plain_c_M' unboxed changed scrutinee Hs Hsz).
  - apply forallb_forall. intros [d ac'] Hda. cbn [snd].
    apply in_flat_map in Hda as ([d0 ac0] & H0 & Hda).
    destruct (rewrite_apply_cont sol en ac0) as [ac1 | ] eqn:Hr;
      [ | destruct Hda ].
    destruct Hda as [Heq' | []]. injection Heq' as <- <-.
    destruct (Hin d0 ac0 H0) as (Hok & Hz' & _).
    exact (proj2 (rewrite_apply_cont_c C en enu ac0 Hci Hok Hz') ac1 Hr).
  - intros [d ac'] Hda. cbn [snd].
    apply in_flat_map in Hda as ([d0 ac0] & H0 & Hda).
    destruct (rewrite_apply_cont sol en ac0) as [ac1 | ] eqn:Hr;
      [ | destruct Hda ].
    destruct Hda as [Heq' | []]. injection Heq' as <- <-.
    destruct (Hin d0 ac0 H0) as (_ & _ & HD0).
    destruct (rewrite_apply_cont_live en ac0 ac1 Hr) as [Hk Hl].
    rewrite Hk. intros Hin'. exact (HD0 _ Hin' Hl).
Qed.

(* ------------------------------------------------------------------------ *)
(* Handlers                                                                 *)
(* ------------------------------------------------------------------------ *)

Definition decide_c (p : (variable * flambda_kind) * bool) : decision :=
  let '((_, kind), keep) := p in Decided (if keep then Keep kind else Delete).

Lemma handler_decisions_decided (is_exn : bool) (params : list variable)
    (md : handler_metadata) (handler : expr) :
  forallb decided
    (normed
       (handler_decisions true sol unboxed is_exn
          (Handler params md handler)))
  = true ->
  normed
    (handler_decisions true sol unboxed is_exn (Handler params md handler))
  = map decide_c
      (combine (with_kinds params (hm_params_kinds md))
         (handler_mask sol is_exn (Handler params md handler))).
Proof.
  unfold normed, handler_decisions, handler_mask. intros H.
  rewrite map_map in *. rewrite forallb_forall in H. apply map_ext_in.
  intros [[x kind] keep] Hin.
  specialize (H _ (in_map _ _ _ Hin)). cbn beta iota in H |- *.
  destruct (unboxed_of_node unboxed (Node_variable x)); [ discriminate H | ].
  destruct (true && keep && negb (is_var_used sol x kind)) eqn:E;
    [ | reflexivity ].
  destruct keep; [ reflexivity | discriminate E ].
Qed.

(* The two rebuilds read a parameter kept with no use as one kept. *)
Lemma params_of_decisions_normed (bound : list (variable * flambda_kind)) :
  ∀ ds : list decision,
    params_of_decisions bound ds = params_of_decisions bound (normed ds).
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | d ds];
    try reflexivity.
  cbn [normed map params_of_decisions]. fold (normed ds). rewrite <- IH.
  destruct d as [[k | ] | | ]; reflexivity.
Qed.

Lemma is_kept_normed (ds : list decision) :
  map is_kept_decision ds = map is_kept_decision (normed ds).
Proof.
  unfold normed. rewrite map_map. apply map_ext.
  intros [[k | ] | | ]; reflexivity.
Qed.

Lemma decisions_subkinds_normed (rw : subkinds_rewrite)
    (params : list (node * param_subkind)) :
  ∀ ds : list decision,
    decisions_subkinds rw params ds = decisions_subkinds rw params (normed ds).
Proof.
  induction params as [ | [n ps] params IH]; intros [ | d ds];
    try reflexivity.
  cbn [normed map decisions_subkinds]. fold (normed ds). rewrite <- IH.
  destruct d as [[k | ] | | ]; reflexivity.
Qed.

Lemma normed_decided (ds : list decision) :
  forallb decided ds = true -> normed ds = ds.
Proof.
  induction ds as [ | d ds IH]; intros H; [ reflexivity | ].
  cbn [forallb] in H. apply andb_true_iff in H as [Hd H].
  cbn [normed map]. fold (normed ds). rewrite (IH H).
  destruct d; [ reflexivity | discriminate Hd | discriminate Hd ].
Qed.

Lemma kept_decide_c (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    map is_kept_decision (map decide_c (combine bound mask)) = mask.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma params_of_decide_c (bound : list (variable * flambda_kind)) :
  ∀ mask : list bool,
    length mask = length bound ->
    params_of_decisions bound (map decide_c (combine bound mask))
    = filter_kept mask bound.
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma filter_kept_fst {A B : Type} (l : list A) :
  ∀ (l' : list B) (mask : list bool),
    length l = length l' ->
    map fst (filter_kept mask (combine l l')) = filter_kept mask l.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  - rewrite IH by lia. reflexivity.
  - destruct keep; cbn; rewrite IH by lia; reflexivity.
Qed.

Lemma filter_kept_snd {A B : Type} (l : list A) :
  ∀ (l' : list B) (mask : list bool),
    length l = length l' ->
    map snd (filter_kept mask (combine l l')) = filter_kept mask l'.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] [ | keep mask] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  - rewrite IH by lia. reflexivity.
  - destruct keep; cbn; rewrite IH by lia; reflexivity.
Qed.

(* The parameters that the two rebuilds keep of a handler of the
   fragment. *)
Lemma handler_params_c (is_exn : bool) (params : list variable)
    (md : handler_metadata) (handler : expr) :
  length params = length (hm_params_kinds md) ->
  forallb decided
    (normed
       (handler_decisions true sol unboxed is_exn
          (Handler params md handler)))
  = true ->
  map fst
    (params_of_decisions (with_kinds params (hm_params_kinds md))
       (handler_decisions true sol unboxed is_exn (Handler params md handler)))
  = filter_kept (handler_mask sol is_exn (Handler params md handler)) params
  ∧ map snd
      (params_of_decisions (with_kinds params (hm_params_kinds md))
         (handler_decisions true sol unboxed is_exn
            (Handler params md handler)))
    = filter_kept (handler_mask sol is_exn (Handler params md handler))
        (hm_params_kinds md)
  ∧ map is_kept_decision
      (handler_decisions true sol unboxed is_exn (Handler params md handler))
    = handler_mask sol is_exn (Handler params md handler).
Proof.
  intros Hlen Hdec.
  rewrite params_of_decisions_normed, is_kept_normed.
  rewrite (handler_decisions_decided is_exn params md handler Hdec).
  assert (Hm : length (handler_mask sol is_exn (Handler params md handler))
               = length (with_kinds params (hm_params_kinds md)))
    by (unfold handler_mask; apply params_to_keep_length).
  rewrite (params_of_decide_c _ _ Hm), (kept_decide_c _ _ Hm).
  rewrite (with_kinds_combine params _ Hlen).
  split; [ apply filter_kept_fst, Hlen | ].
  split; [ apply filter_kept_snd, Hlen | reflexivity ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Decisions of code                                                        *)
(* ------------------------------------------------------------------------ *)

Lemma decisions_u_decided (nodes : list node) (kinds : list flambda_kind) :
  forallb decided (decisions_u sol unboxed nodes kinds) = true ->
  decisions_u sol unboxed nodes kinds
  = map Decided (decisions_of sol nodes kinds).
Proof.
  unfold decisions_u, decisions_of. intros H. rewrite map_map.
  rewrite forallb_forall in H. apply map_ext_in. intros [n k] Hin.
  specialize (H _ (in_map _ _ _ Hin)). cbn beta iota in H |- *.
  destruct (unboxed_of_node unboxed n); [ discriminate H | reflexivity ].
Qed.

Lemma code_change_decided (id : code_id) (md : code_metadata)
    (pd rd : list decision) :
  code_change_u sol cannot_change code_deps unit_code_ids unboxed id md
  = Changing_u pd rd ->
  forallb decided pd = true -> forallb decided rd = true ->
  ∃ (cd : code_dep) (pd_cc rd_cc : list param_decision),
    lookup id code_deps = Some cd
    ∧ code_calling_convention_change sol cannot_change code_deps
        unit_code_ids id md
      = Changing_calling_convention pd_cc rd_cc
    ∧ pd = map Decided pd_cc ∧ rd = map Decided rd_cc.
Proof.
  unfold code_change_u, code_calling_convention_change.
  destruct (cannot_change_calling_convention cannot_change unit_code_ids id);
    [ discriminate | ].
  destruct (lookup id code_deps) as [cd | ]; [ | discriminate ].
  intros H Hpd Hrd. injection H as <- <-.
  exists cd, (decisions_of sol (cd_params cd)
                (unarize (cm_params_arity md))),
    (decisions_of sol (cd_return cd) (unarize (cm_result_arity md))).
  split; [ reflexivity | split; [ reflexivity | ] ].
  split; apply decisions_u_decided; assumption.
Qed.

(* ------------------------------------------------------------------------ *)
(* No wrapper                                                               *)
(* ------------------------------------------------------------------------ *)

Lemma wrapper_args_u_decided (tl : bool) (rd : list param_decision) :
  ∀ (ks : list flambda_kind) (ads : list decision) (next : positive),
    length rd <= length ks ->
    forallb2 same_decision (normed ads) (map Decided rd) = true
    ∨ (ads = [] ∧ forallb keeps (map Decided rd) = true) ->
    wrapper_args_u tl ads
      (freshen_decisions (combine (map Decided rd) ks) next)
    = Some
        (map (fun p => Var (fst p))
           (wrapper_params
              (freshen_decisions (combine (map Decided rd) ks) next))).
Proof.
  induction rd as [ | d rd IH]; intros ks ads next Hlen Hads.
  - destruct ks; reflexivity.
  - destruct ks as [ | k ks]; cbn [length] in Hlen; [ lia | ].
    cbn [map combine].
    destruct d as [kd | ]; cbn [freshen_decisions].
    + unfold wrapper_params. cbn [flat_map].
      fold (wrapper_params
              (freshen_decisions (combine (map Decided rd) ks)
                 (Pos.succ next))).
      destruct Hads as [Hads | (-> & Hkeeps)].
      * destruct ads as [ | ad ads]; [ discriminate Hads | ].
        cbn [forallb2 normed map] in Hads.
        apply andb_true_iff in Hads as [Had Hads].
        destruct ad as [[k' | ] | | ]; cbn [norm_decision same_decision] in Had;
          try discriminate Had;
          cbn [wrapper_args_u];
          rewrite (IH ks ads (Pos.succ next) ltac:(lia) (or_introl Hads));
          reflexivity.
      * cbn [forallb keeps] in Hkeeps. cbn [wrapper_args_u].
        rewrite (IH ks [] (Pos.succ next) ltac:(lia)
                   (or_intror (conj eq_refl Hkeeps))).
        reflexivity.
    + unfold wrapper_params. cbn [flat_map app].
      fold (wrapper_params
              (freshen_decisions (combine (map Decided rd) ks) next)).
      destruct Hads as [Hads | (-> & Hkeeps)];
        [ | cbn in Hkeeps; discriminate Hkeeps ].
      destruct ads as [ | ad ads]; [ discriminate Hads | ].
      cbn [forallb2 normed map] in Hads.
      apply andb_true_iff in Hads as [Had Hads].
      destruct ad as [[k' | ] | | ]; cbn [norm_decision same_decision] in Had;
        try discriminate Had.
      cbn [wrapper_args_u].
      rewrite (IH ks ads next ltac:(lia) (or_introl Hads)).
      reflexivity.
Qed.

Lemma args_are_params_vars (ps : list (variable * flambda_kind)) :
  args_are_params (map (fun p => Var (fst p)) ps) (map fst ps) = true.
Proof.
  induction ps as [ | [x k] ps IH]; [ reflexivity | ].
  cbn. rewrite Pos.eqb_refl. exact IH.
Qed.

Lemma wrapper_u_none (tl : bool) (next : positive) (enu : unbox_env)
    (ap : apply_expr) (rd : list param_decision) (ks : list flambda_kind)
    (wsk : list param_subkind) :
  length rd <= length ks ->
  match lookup (ap_continuation ap) (ue_decisions enu) with
  | Some ds => forallb2 same_decision (normed ds) (map Decided rd) = true
  | None => forallb keeps (map Decided rd) = true
  end ->
  make_apply_wrapper_u tl next types wsk enu ap
    (combine (map Decided rd) ks) false
  = Apply ap.
Proof.
  intros Hlen Hds. unfold make_apply_wrapper_u. cbv zeta.
  rewrite andb_false_r. cbn iota.
  rewrite (wrapper_args_u_decided tl rd ks _ next Hlen).
  - rewrite args_are_params_vars. reflexivity.
  - destruct (lookup (ap_continuation ap) (ue_decisions enu));
      [ left; exact Hds | right; split; [ reflexivity | exact Hds ] ].
Qed.

Lemma forallb_kept_combine (rd : list param_decision) :
  ∀ ks : list flambda_kind,
    length rd <= length ks ->
    forallb (fun '(keep, (d, _)) => Bool.eqb keep (is_kept d))
      (combine (map is_kept rd) (combine rd ks))
    = true.
Proof.
  induction rd as [ | d rd IH]; intros [ | k ks] Hlen; cbn in Hlen |- *;
    try lia; try reflexivity.
  rewrite Bool.eqb_reflx. apply IH. lia.
Qed.

Lemma forallb_true_combine (rd : list param_decision) :
  ∀ ks : list flambda_kind,
    forallb is_kept rd = true -> length rd <= length ks ->
    forallb (fun '(keep, (d, _)) => Bool.eqb keep (is_kept d))
      (combine (repeat true (length rd)) (combine rd ks))
    = true.
Proof.
  induction rd as [ | d rd IH]; intros [ | k ks] Hk Hlen;
    cbn in Hlen, Hk |- *; try lia; try reflexivity.
  apply andb_true_iff in Hk as [Hd Hk]. rewrite Hd. apply IH; [ exact Hk | ].
  lia.
Qed.

Lemma wrapper_cc_none (next : positive) (en : rebuild_env) (ap : apply_expr)
    (rd : list param_decision) (ks : list flambda_kind) :
  length rd <= length ks ->
  match lookup (ap_continuation ap) (env_conts en) with
  | Some mask => mask = map is_kept rd
  | None => forallb is_kept rd = true
  end ->
  make_apply_wrapper_cc next en ap (combine rd ks) = Apply ap.
Proof.
  intros Hlen Hm. unfold make_apply_wrapper_cc, apply_decisions.
  rewrite length_combine, (Nat.min_l _ _ Hlen).
  destruct (lookup (ap_continuation ap) (env_conts en)) as [mask | ].
  - subst mask. rewrite firstn_app, length_map, Nat.sub_diag, firstn_O,
      app_nil_r, firstn_all2 by (rewrite length_map; lia).
    rewrite (forallb_kept_combine rd ks Hlen). reflexivity.
  - cbn [app]. rewrite firstn_all2 by (rewrite repeat_length; lia).
    rewrite (forallb_true_combine rd ks Hm Hlen). reflexivity.
Qed.

Lemma kept_same (ds : list decision) (rd : list param_decision) :
  forallb2 same_decision ds (map Decided rd) = true ->
  map is_kept_decision ds = map is_kept rd.
Proof.
  revert rd. induction ds as [ | d ds IH]; intros [ | r rd] H;
    cbn in H |- *; try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [Hd H].
  destruct d as [[k | ] | | ], r as [k' | ]; try discriminate Hd;
    rewrite (IH rd H); reflexivity.
Qed.

Lemma keeps_kept (rd : list param_decision) :
  forallb keeps (map Decided rd) = true -> forallb is_kept rd = true.
Proof.
  induction rd as [ | [k | ] rd IH]; cbn; intros H;
    [ reflexivity | exact (IH H) | discriminate H ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma group_by_sizes_in {A : Type} (sizes : list nat) :
  ∀ (l g : list A) (a : A), g ∈ group_by_sizes sizes l -> a ∈ g -> a ∈ l.
Proof.
  induction sizes as [ | n sizes IH]; intros l g a Hg Ha; [ destruct Hg | ].
  rewrite <- (firstn_skipn n l). apply in_or_app.
  destruct Hg as [<- | Hg]; [ left; exact Ha | right; exact (IH _ g a Hg Ha) ].
Qed.

Lemma get_args_with_kinds_c (en : rebuild_env) (args : list simple) :
  forallb (plain_simple unboxed) args = true ->
  ∀ ds : list param_decision,
    get_args_with_kinds_u sol unboxed en (map Decided ds) args
    = Some (get_args_with_kinds sol en ds args).
Proof.
  induction args as [ | a args IH]; intros Hargs [ | d ds]; try reflexivity.
  cbn [forallb] in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
  cbn [map get_args_with_kinds_u]. rewrite (IH Hargs ds).
  unfold get_args_with_kinds. cbn [combine flat_map].
  destruct d as [kind | ]; [ | reflexivity ].
  cbn. rewrite (rewrite_simple_u sol unboxed en a Ha). reflexivity.
Qed.

Lemma all_some_groups_c (en : rebuild_env) (l1 : list (list simple)) :
  (∀ g, g ∈ l1 -> forallb (plain_simple unboxed) g = true) ->
  ∀ l2 : list (list param_decision),
    all_some
      (map
         (fun '(args, decisions) =>
            get_args_with_kinds_u sol unboxed en decisions args)
         (combine l1 (map (map Decided) l2)))
    = Some
        (map
           (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
           (combine l1 l2)).
Proof.
  induction l1 as [ | a l1 IH]; intros Hg [ | ds l2]; try reflexivity.
  cbn [map combine].
  change (all_some (?o :: ?l))
    with (match o, all_some l with
          | Some a, Some acc => Some (a :: acc)
          | _, _ => None
          end).
  rewrite (IH (fun g H => Hg g (or_intror H)) l2).
  rewrite (get_args_with_kinds_c en a (Hg a (or_introl eq_refl)) ds).
  reflexivity.
Qed.

Lemma get_args_with_kinds_plain (en : rebuild_env) (args : list simple)
    (ds : list param_decision) :
  forallb (plain_simple MM) args = true ->
  forallb (plain_simple MM) (map fst (get_args_with_kinds sol en ds args))
  = true.
Proof.
  revert ds. induction args as [ | a args IH]; intros [ | d ds] Hargs;
    try reflexivity.
  cbn [forallb] in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
  unfold get_args_with_kinds in *. cbn [combine flat_map].
  destruct d as [kind | ]; [ | exact (IH ds Hargs) ].
  cbn [app map forallb fst]. rewrite (rewrite_plain_M' en a Ha).
  exact (IH ds Hargs).
Qed.

Lemma simples_dead_kinds (en1 en2 : rebuild_env) (l : list simple) :
  env_kinds en1 = env_kinds en2 ->
  existsb (simple_is_dead sol en1) l = existsb (simple_is_dead sol en2) l.
Proof.
  intros H. induction l as [ | s l IH]; [ reflexivity | ].
  cbn [existsb]. rewrite IH. f_equal.
  destruct s; simpl; try reflexivity.
  unfold kind_of_variable. rewrite H. reflexivity.
Qed.

Lemma rewrite_call_kind_c (ap : apply_expr) (id : code_id) :
  code_id_actually_called F sol ap = Some id ->
  rewrite_call_kind F sol pdc unit_code_ids ap = Direct id.
Proof.
  unfold code_id_actually_called, rewrite_call_kind, called_call_kind.
  destruct (ap_call_kind ap); [ | intros H; discriminate H | ];
    destruct (ap_callee ap); try (intros H; discriminate H);
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try (intros H; discriminate H); intros Hs; rewrite Hs; reflexivity.
Qed.

(* A continuation that keeps every parameter, given the results that a
   callee keeps or deletes alike, is given them all. *)
Lemma keeps_same_kept (ds : list decision) (rd : list param_decision) :
  forallb keeps ds = true ->
  forallb2 same_decision ds (map Decided rd) = true ->
  forallb is_kept rd = true.
Proof.
  revert rd. induction ds as [ | d ds IH]; intros [ | r rd] Hk H;
    cbn in H |- *; try discriminate H; [ reflexivity | ].
  cbn [forallb] in Hk. apply andb_true_iff in Hk as [Hd Hk].
  apply andb_true_iff in H as [Hr H].
  destruct d as [[k | ] | | ], r as [k' | ]; try discriminate Hd;
    try discriminate Hr.
  exact (IH rd Hk H).
Qed.

Lemma same_keep_keeps (ds : list decision) (ks : list flambda_kind) :
  forallb2 same_decision ds (map (fun kind => Decided (Keep kind)) ks) = true ->
  forallb keeps ds = true.
Proof.
  revert ks. induction ds as [ | d ds IH]; intros [ | k ks] H;
    cbn in H |- *; try discriminate H; [ reflexivity | ].
  apply andb_true_iff in H as [Hd H].
  destruct d as [[k' | ] | | ]; try discriminate Hd. exact (IH ks H).
Qed.

Lemma keeps_mask (ds : list decision) :
  forallb keeps ds = true -> forallb (fun keep => keep)
                                (map is_kept_decision ds) = true.
Proof.
  induction ds as [ | d ds IH]; intros H; [ reflexivity | ].
  cbn [forallb] in H. apply andb_true_iff in H as [Hd H].
  destruct d as [[k | ] | | ]; try discriminate Hd. exact (IH H).
Qed.

Lemma combine_keep (ks : list flambda_kind) :
  combine (map Decided (map Keep ks)) ks
  = map (fun kind => (Decided (Keep kind), kind)) ks.
Proof.
  induction ks as [ | k ks IH]; [ reflexivity | ]. cbn. rewrite IH.
  reflexivity.
Qed.

Lemma mask_arguments_plain (en : rebuild_env) (args : list simple) :
  forallb (plain_simple MM) args = true ->
  ∀ mask : list bool,
    forallb (plain_simple MM) (mask_arguments en mask args) = true.
Proof.
  induction args as [ | a args IH]; intros Hargs [ | b mask];
    try reflexivity;
    cbn [forallb] in Hargs; apply andb_true_iff in Hargs as [Ha Hargs];
    cbn [mask_arguments forallb]; rewrite (IH Hargs).
  - rewrite Ha. reflexivity.
  - destruct b; [ rewrite Ha; reflexivity | ].
    unfold poison_argument. destruct a; reflexivity.
Qed.

(* The first rebuild makes no wrapper of a call of code that keeps its
   calling convention whose continuation keeps every result. *)
Lemma make_apply_wrapper_none (en : rebuild_env) (ap : apply_expr) :
  match lookup (ap_continuation ap) (env_conts en) with
  | Some mask => forallb (fun keep => keep) mask = true
  | None => True
  end ->
  make_apply_wrapper ff types en ap = Apply ap.
Proof.
  intros H. unfold make_apply_wrapper.
  destruct (lookup (ap_continuation ap) (env_conts en)); [ | reflexivity ].
  rewrite H. reflexivity.
Qed.

Lemma kept_mask_none (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (k : continuation)
    (ks : list flambda_kind) :
  cinv C en enu ->
  returns_ok C k (map (fun kind => Decided (Keep kind)) ks) = true ->
  match lookup k (env_conts en) with
  | Some mask => forallb (fun keep => keep) mask = true
  | None => True
  end
  ∧ match lookup k (ue_decisions enu) with
    | Some ds =>
      forallb2 same_decision (normed ds) (map Decided (map Keep ks)) = true
    | None => forallb keeps (map Decided (map Keep ks)) = true
    end.
Proof.
  intros Hci Hret. unfold returns_ok in Hret.
  rewrite map_map.
  destruct (lookup k C) as [[ds [ | ]] | ] eqn:HC; [ | discriminate Hret | ].
  - destruct (ci_app _ _ _ Hci _ _ _ HC (or_introl eq_refl))
      as ((dsu & Hd & Hn) & Hm & _).
    rewrite Hd. split; [ | cbn beta iota; rewrite Hn; exact Hret ].
    destruct Hm as [Hm | [Hm _]]; rewrite Hm; [ | exact I ].
    exact (keeps_mask ds (same_keep_keeps ds ks Hret)).
  - destruct (ci_out _ _ _ Hci _ HC) as (Hd & Hm). rewrite Hd, Hm.
    split; [ exact I | exact Hret ].
Qed.

(* A parameter kept with no use is given what is given, as an extra
   argument too. *)
Lemma get_extra_args_normed (en : rebuild_env)
    (args : list (simple * flambda_kind)) :
  ∀ ds : list decision,
    get_extra_args sol unboxed en ds args
    = get_extra_args sol unboxed en (normed ds) args.
Proof.
  induction args as [ | [s k] args IH]; intros [ | d ds]; try reflexivity.
  unfold normed in *. cbn [map get_extra_args]. rewrite (IH ds).
  destruct (get_extra_args sol unboxed en (map norm_decision ds) args);
    [ | reflexivity ].
  destruct d as [[k' | ] | | ]; reflexivity.
Qed.

Lemma get_extra_args_decided (en : rebuild_env)
    (args : list (simple * flambda_kind)) :
  forallb (plain_simple unboxed) (map fst args) = true ->
  ∀ ds : list decision,
    forallb decided ds = true ->
    get_extra_args sol unboxed en ds args
    = Some (filter_kept (map is_kept_decision ds)
              (map (fun '(s, kind) => (rewrite_simple sol en s, kind)) args)).
Proof.
  induction args as [ | [s k] args IH]; intros Hargs ds Hds;
    [ destruct ds; reflexivity | ].
  cbn [map fst forallb] in Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
  destruct ds as [ | d ds].
  - cbn [get_extra_args]. rewrite (IH Hargs [] eq_refl).
    rewrite (rewrite_simple_u sol unboxed en s Hs). reflexivity.
  - cbn [forallb] in Hds. apply andb_true_iff in Hds as [Hd Hds].
    cbn [get_extra_args]. rewrite (IH Hargs ds Hds).
    destruct d as [[kind | ] | fields | kind']; cbn in Hd; try discriminate Hd;
      cbn; rewrite ?(rewrite_simple_u sol unboxed en s Hs); reflexivity.
Qed.

Lemma extra_plain (en : rebuild_env) (extra : list (simple * flambda_kind)) :
  forallb (plain_simple MM) (map fst extra) = true ->
  ∀ mask : list bool,
    forallb (plain_simple MM)
      (map fst
         (filter_kept mask
            (map (fun '(s, kind) => (rewrite_simple sol en s, kind)) extra)))
    = true.
Proof.
  induction extra as [ | [s k] extra IH]; intros H mask;
    [ destruct mask; reflexivity | ].
  cbn [map fst forallb] in H. apply andb_true_iff in H as [Hs H].
  destruct mask as [ | b mask]; cbn [filter_kept map].
  - cbn [map fst forallb]. rewrite (rewrite_plain_M' en s Hs).
    exact (IH H []).
  - destruct b; cbn [map fst forallb];
      [ rewrite (rewrite_plain_M' en s Hs) | ]; exact (IH H mask).
Qed.

(* The extra arguments of the exception continuation of a call of the
   fragment: the same in both rebuilds, plain. *)
Lemma extra_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ap : apply_expr) :
  cinv C en enu -> extra_args_ok unboxed changed C ap = true ->
  (∀ z, z ∈ variables_in_simples (exn_extra_simples ap) -> fresh z) ->
  rebuild_exn_extra_args_u sol unboxed enu ap
  = Some (rebuild_exn_extra_args sol en ap)
  ∧ forallb (plain_simple MM) (map fst (rebuild_exn_extra_args sol en ap))
    = true.
Proof.
  intros Hci Hok Hz. unfold extra_args_ok in Hok.
  apply andb_true_iff in Hok as [Hplain Hk].
  assert (Hstrong := plain_cs_strong _ Hplain Hz).
  assert (HplainM := plain_cs_M' _ Hplain Hz).
  unfold exn_extra_simples in Hstrong, HplainM.
  assert (Hrw : ∀ ds, forallb decided ds = true ->
            get_extra_args sol unboxed (ue_env enu) ds (ap_exn_extra_args ap)
            = Some (filter_kept (map is_kept_decision ds)
                      (map (fun '(s, kind) => (rewrite_simple sol en s, kind))
                         (ap_exn_extra_args ap)))).
  { intros ds Hds.
    rewrite (get_extra_args_decided (ue_env enu) _ Hstrong ds Hds).
    do 2 f_equal. apply map_ext. intros [s k].
    rewrite (rewrite_simple_kinds sol (ue_env enu) en s (ci_kinds _ _ _ Hci)).
    reflexivity. }
  split.
  2: { unfold rebuild_exn_extra_args.
       destruct (lookup (ap_exn_continuation ap) (env_conts en));
         [ apply extra_plain; exact HplainM | ].
       pose proof (extra_plain en _ HplainM []) as H.
       rewrite (filter_kept_keeps [] eq_refl) in H. exact H. }
  unfold rebuild_exn_extra_args_u, rebuild_exn_extra_args.
  destruct (ap_exn_extra_args ap) as [ | x xs] eqn:Hx.
  - destruct (lookup (ap_exn_continuation ap) (ue_decisions enu));
      destruct (lookup (ap_exn_continuation ap) (env_conts en));
      reflexivity.
  - destruct (lookup (ap_exn_continuation ap) C) as [[ds [ | ]] | ] eqn:HC;
      try discriminate Hk.
    destruct (ci_app _ _ _ Hci _ _ _ HC (or_introl eq_refl))
      as ((dsu & Hd & Hn) & Hm & Hdec).
    assert (Htl : normed (tl dsu) = tl ds)
      by (rewrite <- Hn; destruct dsu; reflexivity).
    assert (Hdtl : forallb decided (tl ds) = true).
    { destruct ds as [ | d ds]; [ reflexivity | ].
      cbn [forallb] in Hdec. apply andb_true_iff in Hdec as [_ Hdec].
      exact Hdec. }
    rewrite Hd. cbv beta iota.
    rewrite get_extra_args_normed, Htl, (Hrw (tl ds) Hdtl).
    destruct Hm as [Hm | [Hm Hkeeps]]; rewrite Hm; cbv beta iota.
    + destruct ds; reflexivity.
    + rewrite (filter_kept_keeps (tl ds)); [ reflexivity | ].
      destruct ds as [ | d ds]; [ reflexivity | ].
      cbn [forallb] in Hkeeps. apply andb_true_iff in Hkeeps as [_ Hkeeps].
      exact Hkeeps.
Qed.

(* A call of the fragment is rebuilt alike, with no wrapper; or, of code
   that never returns, returning in the second run to a handler
   [Invalid]. *)
Lemma call_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ap : apply_expr)
    (D : list continuation) :
  cinv C en enu ->
  repr_call_w F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  dead_in D (live (Apply ap)) ->
  rr MM CH CSf RCHf D (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hcall Hz Hff Hffu HD. unfold repr_call_w in Hcall.
  destruct (ap_callee ap) as [f | | ] eqn:Hcallee; try discriminate Hcall.
  destruct (code_id_actually_called F sol ap) as [id | ] eqn:Hcalled;
    [ | discriminate Hcall ].
  apply andb_true_iff in Hcall as [Hkc Hcall].
  destruct (lookup id codes) as [md | ] eqn:Hmd; [ | discriminate Hcall ].
  destruct (lookup id my_closures) as [mc | ] eqn:Hmc;
    [ | discriminate Hcall ].
  apply andb_true_iff in Hcall as [Hcall Hch].
  apply andb_true_iff in Hcall as [Hcall Hextra].
  apply andb_true_iff in Hcall as [Hcall Hargs].
  apply andb_true_iff in Hcall as [Hcall Hra].
  apply andb_true_iff in Hcall as [Hcall Haa].
  apply andb_true_iff in Hcall as [Hcall Hfu].
  apply andb_true_iff in Hcall as [Hnr Hcl].
  apply arity_eqb_eq in Haa, Hra.
  unfold variables_in_apply in Hz. rewrite Hcallee in Hz.
  assert (Hfz : fresh f)
    by (apply Hz, in_or_app; left; left; reflexivity).
  assert (Hargz : ∀ z, z ∈ variables_in_simples (ap_args ap) -> fresh z)
    by (intros z H; apply Hz, in_or_app; right; apply in_or_app; left;
        exact H).
  assert (Hstrong := plain_cs_strong (ap_args ap) Hargs Hargz).
  assert (Hplain := plain_cs_M' (ap_args ap) Hargs Hargz).
  assert (Hfnu : unboxed_of_node unboxed (Node_variable f) = None)
    by (destruct (fields_of unboxed f) eqn:E; [ discriminate Hfu | exact E ]).
  set (ci := Code_info (CH mc) (length (cm_params_arity md)) (nr id)).
  assert (HCS : CSf id = Some ci)
    by (unfold CSf; rewrite Hmd, Hmc; reflexivity).
  assert (Hcok : callee_ok MM CH ci (Var f)).
  { unfold callee_ok. cbn [ci_my_closure ci].
    apply repr_option_eqb_eq in Hcl. unfold changed_name in Hcl.
    rewrite (CHf_var unboxed changed mc (my_closures_fresh id mc Hmc)),
      <- Hcl.
    destruct (changed_of_node changed (Node_variable f)) as [r | ] eqn:Hf.
    - exists f. split; [ reflexivity | ].
      rewrite (CHf_var unboxed changed f Hfz). exact Hf.
    - cbn [plain_simple]. rewrite plain_variable_M', Hf, andb_true_r.
      unfold plain_variable. rewrite Hfz. unfold Keyed.is_bound.
      destruct (fields_of unboxed f) eqn:E; [ discriminate Hfu | ].
      unfold fields_of in E. rewrite E. reflexivity. }
  assert (Hxz : ∀ z, z ∈ variables_in_simples (exn_extra_simples ap) ->
                     fresh z)
    by (intros z H; apply Hz, in_or_app; right; apply in_or_app; right;
        exact H).
  destruct (extra_rebuilt C en enu ap Hci Hextra Hxz) as [Hxu Hxplain].
  assert (Hlive_k : apply_never_returns F sol code_deps codes ap = false ->
                    ap_continuation ap ∉ D).
  { intros Hnr_ap Hin. apply (HD _ Hin). cbn [live_conts]. rewrite Hnr_ap.
    left. reflexivity. }
  assert (Hlive_e : ap_exn_continuation ap ∉ D).
  { intros Hin. apply (HD _ Hin). cbn [live_conts].
    destruct (apply_never_returns F sol code_deps codes ap);
      [ left; reflexivity | right; left; reflexivity ]. }
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
              md)
    as [ | pd rd] eqn:Hchange.
  - (* Code that keeps its calling convention: the arguments at the same
       mask in both, all the results expected. *)
    apply andb_true_iff in Hch as [Hkept Hret].
    assert (Hncc : code_calling_convention_change sol cannot_change
                     code_deps unit_code_ids id md
                   = Not_changing_calling_convention).
    { unfold code_change_u in Hchange.
      unfold code_calling_convention_change.
      destruct (cannot_change_calling_convention cannot_change unit_code_ids
                  id);
        [ reflexivity | ].
      destruct (lookup id code_deps); [ discriminate Hchange | reflexivity ].
      }
    unfold kept_args_ok in Hkept.
    destruct (kept_code_mask F sol code_deps codes ap) as [mask_u | ]
      eqn:Hmu; [ | discriminate Hkept ].
    unfold masked_args_ok in Hkept.
    apply andb_true_iff in Hkept as [Hmeq _].
    apply FlambdaReaperUnboxingRefinesCallsProofs.forallb2_eqb_eq in Hmeq.
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    unfold rebuild_apply_cc, apply_calling_convention_change.
    rewrite Hcalled. unfold get_calling_convention_change.
    rewrite Hmd, Hncc.
    unfold rebuild_apply, rebuild_apply_u.
    rewrite (simples_dead_kinds (ue_env enu) en _ (ci_kinds _ _ _ Hci)).
    rewrite Hcallee.
    replace (simple_is_dead sol (ue_env enu) (Var f))
      with (simple_is_dead sol en (Var f))
      by (cbn; unfold kind_of_variable; rewrite (ci_kinds _ _ _ Hci);
          reflexivity).
    destruct (simple_is_dead sol en (Var f)
              || existsb (simple_is_dead sol en) (ap_args ap));
      [ constructor | ].
    rewrite Hxu.
    unfold apply_change_u. rewrite Hcalled, Hmd, Hchange.
    cbn [unboxed_of]. rewrite Hfnu.
    unfold kept_arguments_u. cbn beta iota. rewrite Hmu. cbn beta iota.
    unfold rebuild_arguments. rewrite Hcallee.
    cbn [node_of_simple].
    assert (Hmask : mask_u
                    = match ap_call_kind ap with
                      | Direct _ | Indirect_known_arity _ =>
                        arguments_used_by_known_arity_call F sol
                          (Node_variable f) (ap_args ap)
                      | Indirect_unknown_arity =>
                        arguments_used_by_unknown_arity_call F sol
                          (Node_variable f) (ap_args_arity ap)
                      end).
    { destruct (ap_call_kind ap); try exact (eq_sym Hmeq).
      discriminate Hkc. }
    rewrite <- Hmask.
    rewrite (FlambdaReaperUnboxingRefinesCallsProofs.mask_arguments_kinds
               (ue_env enu) en (ap_args ap) (ci_kinds _ _ _ Hci)).
    set (ap1 := {| ap_callee := Var f;
                   ap_call_kind :=
                     rewrite_call_kind F sol pdc unit_code_ids ap;
                   ap_args := mask_arguments en mask_u (ap_args ap);
                   ap_continuation := ap_continuation ap;
                   ap_exn_continuation := ap_exn_continuation ap;
                   ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
                   ap_args_arity := ap_args_arity ap;
                   ap_return_arity := ap_return_arity ap |}).
    assert (Hok : call_ok_c MM CH CSf ap1 id ci).
    { split; [ exact (rewrite_call_kind_c ap id Hcalled) | ].
      split; [ exact HCS | ].
      split; [ exact Hcok | ].
      split; [ exact (mask_arguments_plain en _ Hplain mask_u) | ].
      split; [ exact Hxplain | ].
      cbn [ap_args_arity ap1 ci_nparams ci]. rewrite Haa. reflexivity. }
    assert (Hnr_ap : apply_never_returns F sol code_deps codes ap
                     = code_never_returns sol code_deps id)
      by (unfold apply_never_returns; rewrite Hcalled, Hmd; reflexivity).
    rewrite Hnr_ap.
    destruct (code_never_returns sol code_deps id) eqn:Hcnr.
    + (* A call of code that never returns. *)
      cbn [negb orb] in Hnr.
      assert (Hnok : never_ok MM CH CSf RCHf ap1)
        by (left; exists id, ci; split; assumption).
      unfold make_apply_wrapper_u. cbv zeta. cbn [andb].
      unfold make_apply_wrapper.
      destruct (lookup (ap_continuation ap1) (env_conts en)) as [mask | ];
        [ destruct (forallb (fun keep : bool => keep) mask) | ].
      * apply (rr_call_never_r MM CH CSf RCHf D ap1); try assumption.
        intros Heq. exact (Hffu (eq_sym Heq)).
      * match goal with
        | |- rr _ _ _ _ _ (Let_cont_non_recursive ?wc (Handler ?cps ?cmd ?hb) _)
               (Let_cont_non_recursive ?w (Handler ?wps ?wmd Invalid) _) =>
          change (Apply (with_continuation ap1 ffu))
            with (Apply (with_continuation (with_continuation ap1 ff) ffu));
          apply (rr_call_never_wraps MM CH CSf RCHf D (with_continuation ap1 ff)
                   wc cps cmd hb w wps wmd)
        end; try assumption.
        -- intros Heq. exact (Hff (eq_sym Heq)).
        -- intros Heq. exact (Hffu (eq_sym Heq)).
      * apply (rr_call_never_r MM CH CSf RCHf D ap1); try assumption.
        intros Heq. exact (Hffu (eq_sym Heq)).
    + cbn [orb] in Hret.
      unfold kept_results in Hret. rewrite <- Hra in Hret.
      destruct (kept_mask_none C en enu (ap_continuation ap)
                  (unarize (ap_return_arity ap)) Hci Hret)
        as [Hm Hd].
      rewrite (make_apply_wrapper_none en ap1 Hm).
      rewrite <- combine_keep.
      rewrite (wrapper_u_none true ffu enu ap1
                 (map Keep (unarize (ap_return_arity ap)))
                 (unarize (ap_return_arity ap)) _
                 (Nat.eq_le_incl _ _ (length_map _ _)) Hd).
      destruct Hok as (Hk1 & Hk2 & Hk3 & Hk4 & Hk5 & Hk6).
      apply (rr_call MM CH CSf RCHf D ap1 id ci); try assumption.
      exact (Hlive_k Hnr_ap).
  - (* Code that changes its calling convention. *)
    apply andb_true_iff in Hch as [Hch Hret].
    apply andb_true_iff in Hch as [Hch Hrlen].
    apply andb_true_iff in Hch as [Hch Hmcd].
    apply andb_true_iff in Hch as [Hpd Hrd].
    destruct (code_change_decided id md pd rd Hchange Hpd Hrd)
      as (cd & pd_cc & rd_cc & Hcd & Hcc & -> & ->).
    apply Nat.eqb_eq in Hrlen. rewrite length_map in Hrlen.
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    unfold rebuild_apply_cc, apply_calling_convention_change.
    rewrite Hcalled. unfold get_calling_convention_change. rewrite Hmd, Hcc.
    unfold rebuild_apply_u.
    rewrite (simples_dead_kinds (ue_env enu) en _ (ci_kinds _ _ _ Hci)).
    rewrite Hcallee.
    replace (simple_is_dead sol (ue_env enu) (Var f))
      with (simple_is_dead sol en (Var f))
      by (cbn; unfold kind_of_variable; rewrite (ci_kinds _ _ _ Hci);
          reflexivity).
    destruct (simple_is_dead sol en (Var f)
              || existsb (simple_is_dead sol en) (ap_args ap));
      [ constructor | ].
    rewrite Hxu.
    unfold apply_change_u. rewrite Hcalled, Hmd, Hchange.
    unfold apply_my_closure_u. rewrite Hcalled, Hmd.
    destruct (my_closure_decision_u cannot_change code_deps unit_code_ids
                unboxed id);
      [ discriminate Hmcd | ].
    rewrite group_by_parameter_map'.
    rewrite (all_some_groups_c (ue_env enu)
               (group_by_parameter (ap_args_arity ap) (ap_args ap))
               (fun g Hg =>
                  proj2 (forallb_forall _ g)
                    (fun s Hs =>
                       proj1 (forallb_forall _ (ap_args ap)) Hstrong s
                         (group_by_sizes_in _ _ g s Hg Hs)))).
    rewrite prepend_nil'.
    assert (Hgroups :
      map (fun '(args, decisions) =>
             get_args_with_kinds sol (ue_env enu) decisions args)
        (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
           (group_by_parameter (ap_args_arity ap) pd_cc))
      = map (fun '(args, decisions) =>
               get_args_with_kinds sol en decisions args)
          (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
             (group_by_parameter (ap_args_arity ap) pd_cc))).
    { apply map_ext. intros [args ds]. unfold get_args_with_kinds.
      apply flat_map_ext. intros [a [k | ]]; [ | reflexivity ].
      rewrite (rewrite_simple_kinds sol (ue_env enu) en a
                 (ci_kinds _ _ _ Hci)).
      reflexivity. }
    rewrite Hgroups.
    set (groups := map (fun '(args, decisions) =>
                          get_args_with_kinds sol en decisions args)
                     (combine
                        (group_by_parameter (ap_args_arity ap) (ap_args ap))
                        (group_by_parameter (ap_args_arity ap) pd_cc))).
    cbn [unboxed_of]. rewrite Hfnu.
    rewrite (arity_of_decisions_decided rd_cc).
    set (ap' := {| ap_callee := Var f;
                   ap_call_kind :=
                     rewrite_call_kind F sol pdc unit_code_ids ap;
                   ap_args := flat_map (map fst) groups;
                   ap_continuation := ap_continuation ap;
                   ap_exn_continuation := ap_exn_continuation ap;
                   ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
                   ap_args_arity := arity_of_groups groups;
                   ap_return_arity := arity_of_decisions rd_cc |}).
    assert (Hlen_ks : length rd_cc <= length (unarize (ap_return_arity ap)))
      by (rewrite Hra, Hrlen; lia).
    assert (Hok : call_ok_c MM CH CSf ap' id ci).
    { split; [ exact (rewrite_call_kind_c ap id Hcalled) | ].
      split; [ exact HCS | ].
      split; [ exact Hcok | ].
      split.
      { cbn [ap_args ap']. apply forallb_forall. intros s Hs.
        apply in_flat_map in Hs as (g & Hg & Hs). unfold groups in Hg.
        apply in_map_iff in Hg as ([args ds] & <- & Hin).
        apply in_combine_l in Hin.
        assert (Hpa : forallb (plain_simple MM) args = true).
        { apply forallb_forall. intros a Ha.
          exact (proj1 (forallb_forall _ _) Hplain a
                   (group_by_sizes_in _ _ args a Hin Ha)). }
        exact (proj1 (forallb_forall _ _)
                 (get_args_with_kinds_plain en args ds Hpa) s Hs). }
      split; [ exact Hxplain | ].
      cbn [ap_args_arity ap' ci_nparams ci]. unfold arity_of_groups, groups.
      rewrite !length_map, length_combine, !group_by_parameter_length',
        Nat.min_id, Haa.
      reflexivity. }
    destruct (code_never_returns sol code_deps id) eqn:Hcnr.
    { (* A call of code that never returns. *)
      cbn [negb orb] in Hnr.
      assert (Hnok : never_ok MM CH CSf RCHf ap')
        by (left; exists id, ci; split; assumption).
      assert (Hnr_ap : apply_never_returns F sol code_deps codes ap = true)
        by (unfold apply_never_returns; rewrite Hcalled, Hmd; exact Hcnr).
      rewrite Hnr_ap. unfold make_apply_wrapper_u. cbv zeta. cbn [andb].
      unfold make_apply_wrapper_cc. cbv zeta.
      match goal with
      | |- rr _ _ _ _ _ (if ?b then _ else _) _ => destruct b
      end.
      - apply (rr_call_never_r MM CH CSf RCHf D ap'); try assumption.
        intros Heq. exact (Hffu (eq_sym Heq)).
      - match goal with
        | |- rr _ _ _ _ _ (Let_cont_non_recursive ?wc (Handler ?cps ?cmd ?hb) _)
               (Let_cont_non_recursive ?w (Handler ?wps ?wmd Invalid) _) =>
          change (Apply (with_continuation ap' ffu))
            with (Apply (with_continuation (with_continuation ap' ff) ffu));
          apply (rr_call_never_wraps MM CH CSf RCHf D (with_continuation ap' ff)
                   wc cps cmd hb w wps wmd)
        end; try assumption.
        + intros Heq. exact (Hff (eq_sym Heq)).
        + intros Heq. exact (Hffu (eq_sym Heq)). }
    cbn [negb orb] in Hret.
    assert (Hnr_ap : apply_never_returns F sol code_deps codes ap = false)
      by (unfold apply_never_returns; rewrite Hcalled, Hmd; exact Hcnr).
    unfold returns_ok in Hret.
    rewrite (wrapper_cc_none ff en ap' rd_cc _ Hlen_ks).
    2: { cbn [ap_continuation ap'].
         destruct (lookup (ap_continuation ap) C) as [[ds [ | ]] | ] eqn:HC;
           [ | discriminate Hret | ].
         - destruct (ci_app _ _ _ Hci _ _ _ HC (or_introl eq_refl))
             as (_ & Hm & _).
           destruct Hm as [Hm | [Hm Hkeeps]]; rewrite Hm.
           + exact (kept_same ds rd_cc Hret).
           + exact (keeps_same_kept ds rd_cc Hkeeps Hret).
         - destruct (ci_out _ _ _ Hci _ HC) as (_ & Hm). rewrite Hm.
           exact (keeps_kept rd_cc Hret). }
    rewrite Hnr_ap.
    rewrite (wrapper_u_none true ffu enu ap' rd_cc _ _ Hlen_ks).
    2: { cbn [ap_continuation ap'].
         destruct (lookup (ap_continuation ap) C) as [[ds [ | ]] | ] eqn:HC;
           [ | discriminate Hret | ].
         - destruct (ci_app _ _ _ Hci _ _ _ HC (or_introl eq_refl))
             as ((dsu & Hd & Hn) & _ & _).
           rewrite Hd. cbn beta iota. rewrite Hn. exact Hret.
         - destruct (ci_out _ _ _ Hci _ HC) as (Hd & _). rewrite Hd.
           exact Hret. }
    destruct Hok as (Hk1 & Hk2 & Hk3 & Hk4 & Hk5 & Hk6).
    apply (rr_call MM CH CSf RCHf D ap' id ci); try assumption.
    exact (Hlive_k Hnr_ap).
Qed.

(* The call kind that both rebuilds give a call whose code is not known,
   of a variable: one of known arity stays one, of any list of code ids;
   one of unknown arity stays one. *)
Lemma rewrite_call_kind_ind (ap : apply_expr) (f : variable) :
  ap_callee ap = Var f -> code_id_actually_called F sol ap = None ->
  match ap_call_kind ap with
  | Direct _ => True
  | Indirect_known_arity _ =>
    ∃ ids : option (list code_id),
      rewrite_call_kind F sol pdc unit_code_ids ap = Indirect_known_arity ids
  | Indirect_unknown_arity =>
    rewrite_call_kind F sol pdc unit_code_ids ap = Indirect_unknown_arity
  end.
Proof.
  intros Hcallee Hcalled. unfold code_id_actually_called in Hcalled.
  unfold rewrite_call_kind, called_call_kind.
  destruct (ap_call_kind ap) as [id | | ids0];
    [ exact I | reflexivity | ].
  rewrite Hcallee in Hcalled |- *. cbn [node_of_simple] in Hcalled |- *.
  destruct (code_id_actually_directly_called F sol (Node_variable f))
    as [l | ]; [ | eexists; reflexivity ].
  rewrite Hcalled. destruct (Keyed.is_empty l); eexists; reflexivity.
Qed.

(* A call whose code is not known, of [ind_call_r]: the same call in both
   rebuilds, at the same mask, with no wrapper; or, if it never returns,
   returning in the second rebuild to a handler [Invalid]. *)
Lemma ind_call_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ap : apply_expr) (f : variable)
    (D : list continuation) :
  cinv C en enu ->
  ap_callee ap = Var f -> code_id_actually_called F sol ap = None ->
  ind_call_r F sol code_deps codes unboxed changed my_closures nr reach C ap
    f
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  dead_in D (live (Apply ap)) ->
  rr MM CH CSf RCHf D (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hcallee Hcalled Hind Hz Hff Hffu HD. unfold ind_call_r in Hind.
  cbv zeta in Hind.
  apply andb_true_iff in Hind as [Hind Hkind].
  apply andb_true_iff in Hind as [Hind Hret].
  apply andb_true_iff in Hind as [Hind Hextra].
  apply andb_true_iff in Hind as [Hind Hmask].
  apply andb_true_iff in Hind as [Hind Hargs].
  apply andb_true_iff in Hind as [Hfu Hfc].
  apply FlambdaReaperUnboxingRefinesCallsProofs.forallb2_eqb_eq in Hmask.
  unfold variables_in_apply in Hz. rewrite Hcallee in Hz.
  assert (Hfz : fresh f)
    by (apply Hz, in_or_app; left; left; reflexivity).
  assert (Hargz : ∀ z, z ∈ variables_in_simples (ap_args ap) -> fresh z)
    by (intros z H; apply Hz, in_or_app; right; apply in_or_app; left;
        exact H).
  assert (Hplain := plain_cs_M' (ap_args ap) Hargs Hargz).
  assert (Hfnu : unboxed_of_node unboxed (Node_variable f) = None)
    by (destruct (fields_of unboxed f) eqn:E; [ discriminate Hfu | exact E ]).
  assert (Hfp : plain_variable MM f = true).
  { rewrite plain_variable_M'. unfold changed_name in Hfc.
    rewrite Hfc, andb_true_r.
    unfold plain_variable. rewrite Hfz. unfold Keyed.is_bound.
    destruct (fields_of unboxed f) eqn:E; [ discriminate Hfu | ].
    unfold fields_of in E. rewrite E. reflexivity. }
  assert (Hxz : ∀ z, z ∈ variables_in_simples (exn_extra_simples ap) ->
                     fresh z)
    by (intros z H; apply Hz, in_or_app; right; apply in_or_app; right;
        exact H).
  destruct (extra_rebuilt C en enu ap Hci Hextra Hxz) as [Hxu Hxplain].
  assert (Hlive_e : ap_exn_continuation ap ∉ D).
  { intros Hin. apply (HD _ Hin). cbn [live_conts].
    destruct (apply_never_returns F sol code_deps codes ap);
      [ left; reflexivity | right; left; reflexivity ]. }
  (* The code of the unit, as calls see it. *)
  assert (Hcode : ∀ (id : code_id) (ci : code_info),
            CSf id = Some ci ->
            ∃ (md : code_metadata) (mc : variable),
              (id, md) ∈ codes ∧ lookup id my_closures = Some mc
              ∧ ci_my_closure ci = CH mc
              ∧ ci_nparams ci = length (cm_params_arity md)
              ∧ ci_never ci = nr id).
  { intros id ci Hci'. unfold CSf in Hci'.
    destruct (lookup id codes) as [md | ] eqn:Hmd; [ | discriminate Hci' ].
    destruct (lookup id my_closures) as [mc | ] eqn:Hmc;
      [ | discriminate Hci' ].
    injection Hci' as <-. exists md, mc.
    split; [ exact (lookup_in _ _ _ Hmd) | ].
    split; [ reflexivity | split; [ reflexivity | split; reflexivity ] ]. }
  assert (Hplain_mc : ∀ (id : code_id) (mc : variable),
            lookup id my_closures = Some mc ->
            plain_my_closure changed my_closures id = true -> CH mc = None).
  { intros id mc Hmc Hp. unfold plain_my_closure in Hp. rewrite Hmc in Hp.
    rewrite (CHf_var unboxed changed mc (my_closures_fresh id mc Hmc)).
    unfold changed_name in Hp.
    destruct (changed_of_node changed (Node_variable mc));
      [ discriminate Hp | reflexivity ]. }
  pose proof (rewrite_call_kind_ind ap f Hcallee Hcalled) as Hrk.
  unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
  unfold rebuild_apply_cc, apply_calling_convention_change.
  rewrite Hcalled.
  unfold rebuild_apply, rebuild_apply_u.
  rewrite (simples_dead_kinds (ue_env enu) en _ (ci_kinds _ _ _ Hci)).
  rewrite Hcallee.
  replace (simple_is_dead sol (ue_env enu) (Var f))
    with (simple_is_dead sol en (Var f))
    by (cbn; unfold kind_of_variable; rewrite (ci_kinds _ _ _ Hci);
        reflexivity).
  destruct (simple_is_dead sol en (Var f)
            || existsb (simple_is_dead sol en) (ap_args ap));
    [ constructor | ].
  rewrite Hxu.
  unfold apply_change_u. rewrite Hcalled.
  cbn [unboxed_of]. rewrite Hfnu.
  unfold kept_arguments_u. cbn beta iota.
  unfold kept_code_mask. rewrite Hcalled. cbn beta iota.
  unfold rebuild_arguments. rewrite Hcallee.
  cbn [node_of_simple].
  set (mask := call_mask F sol f ap).
  assert (Hm1 : (match ap_call_kind ap with
                 | Direct _ | Indirect_known_arity _ =>
                   arguments_used_by_known_arity_call F sol
                     (Node_variable f) (ap_args ap)
                 | Indirect_unknown_arity =>
                   arguments_used_by_unknown_arity_call F sol
                     (Node_variable f) (ap_args_arity ap)
                 end)
                = mask)
    by reflexivity.
  assert (Hm2 : (match ap_call_kind ap with
                 | Direct _ | Indirect_known_arity _ =>
                   arguments_used_by_known_arity_call F sol_u
                     (Node_variable f) (ap_args ap)
                 | Indirect_unknown_arity =>
                   arguments_used_by_unknown_arity_call F sol_u
                     (Node_variable f) (ap_args_arity ap)
                 end)
                = mask)
    by exact (eq_sym Hmask).
  rewrite Hm1, Hm2.
  rewrite (FlambdaReaperUnboxingRefinesCallsProofs.mask_arguments_kinds
             (ue_env enu) en (ap_args ap) (ci_kinds _ _ _ Hci)).
  set (ap1 := {| ap_callee := Var f;
                 ap_call_kind :=
                   rewrite_call_kind F sol pdc unit_code_ids ap;
                 ap_args := mask_arguments en mask (ap_args ap);
                 ap_continuation := ap_continuation ap;
                 ap_exn_continuation := ap_exn_continuation ap;
                 ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
                 ap_args_arity := ap_args_arity ap;
                 ap_return_arity := ap_return_arity ap |}).
  destruct (apply_never_returns F sol code_deps codes ap) eqn:Hnr.
  - (* A call that never returns: of code of [nr]. *)
    assert (Hnok : never_ok MM CH CSf RCHf ap1).
    { right. exists f. split; [ reflexivity | ]. split; [ exact Hfp | ].
      split; [ exact (mask_arguments_plain en _ Hplain mask) | ].
      split; [ exact Hxplain | ].
      unfold ind_never. cbn [ap_call_kind ap1 ap_args_arity].
      destruct (ap_call_kind ap) as [id0 | | ids0] eqn:Hk;
        [ discriminate Hkind | | ].
      - rewrite Hrk.
        apply andb_true_iff in Hkind as [Hone Hall].
        apply Nat.eqb_eq in Hone. split; [ exact Hone | ].
        intros id ci Hr Hci'.
        destruct (Hcode id ci Hci') as (md & mc & Hin & Hmc & -> & -> & ->).
        pose proof (proj1 (forallb_forall _ _) Hall (id, md) Hin) as H.
        cbn beta iota in H. unfold RCHf in Hr. rewrite Hr in H.
        cbn [negb orb] in H.
        apply andb_true_iff in H as [H Hn].
        apply andb_true_iff in H as [Hp _].
        apply andb_true_iff in Hn as [Hn Hl]. apply Nat.eqb_eq in Hl.
        split; [ exact (Hplain_mc id mc Hmc Hp) | split; assumption ].
      - destruct Hrk as (ids & ->).
        intros id ci Hr Hci'.
        destruct (Hcode id ci Hci') as (md & mc & Hin & Hmc & -> & _ & ->).
        pose proof (proj1 (forallb_forall _ _) Hkind (id, md) Hin) as H.
        cbn beta iota in H. unfold RCHf in Hr. rewrite Hr in H.
        cbn [negb orb] in H.
        apply andb_true_iff in H as [Hp Hn].
        split; [ exact (Hplain_mc id mc Hmc Hp) | exact Hn ]. }
    unfold make_apply_wrapper_u. cbv zeta. cbn [andb].
    unfold make_apply_wrapper.
    destruct (lookup (ap_continuation ap1) (env_conts en)) as [mask0 | ];
      [ destruct (forallb (fun keep : bool => keep) mask0) | ].
    + apply (rr_call_never_r MM CH CSf RCHf D ap1); try assumption.
      intros Heq. exact (Hffu (eq_sym Heq)).
    + match goal with
      | |- rr _ _ _ _ _ (Let_cont_non_recursive ?wc (Handler ?cps ?cmd ?hb) _)
             (Let_cont_non_recursive ?w (Handler ?wps ?wmd Invalid) _) =>
        change (Apply (with_continuation ap1 ffu))
          with (Apply (with_continuation (with_continuation ap1 ff) ffu));
        apply (rr_call_never_wraps MM CH CSf RCHf D (with_continuation ap1 ff)
                 wc cps cmd hb w wps wmd)
      end; try assumption.
      * intros Heq. exact (Hff (eq_sym Heq)).
      * intros Heq. exact (Hffu (eq_sym Heq)).
    + apply (rr_call_never_r MM CH CSf RCHf D ap1); try assumption.
      intros Heq. exact (Hffu (eq_sym Heq)).
  - (* A call that returns, to a continuation that keeps every result. *)
    cbn [orb] in Hret.
    assert (Hlive_k : ap_continuation ap ∉ D).
    { intros Hin. apply (HD _ Hin). cbn [live_conts]. rewrite Hnr.
      left. reflexivity. }
    destruct (kept_mask_none C en enu (ap_continuation ap)
                (unarize (ap_return_arity ap)) Hci Hret)
      as [Hm Hd].
    rewrite (make_apply_wrapper_none en ap1 Hm).
    rewrite <- combine_keep.
    rewrite (wrapper_u_none true ffu enu ap1
               (map Keep (unarize (ap_return_arity ap)))
               (unarize (ap_return_arity ap)) _
               (Nat.eq_le_incl _ _ (length_map _ _)) Hd).
    apply (rr_call_ind MM CH CSf RCHf D ap1 f); try assumption;
      [ reflexivity | | exact (mask_arguments_plain en _ Hplain mask) ].
    (* The code that the flow may bring to [f]. *)
    unfold ind_targets. cbn [ap_call_kind ap1 ap_args_arity].
    destruct (ap_call_kind ap) as [id0 | | ids0] eqn:Hk;
      [ discriminate Hkind | | ].
    + rewrite Hrk.
      apply andb_true_iff in Hkind as [Hone Hall].
      apply Nat.eqb_eq in Hone. split; [ exact Hone | ].
      intros id ci Hr Hci'.
      destruct (Hcode id ci Hci') as (md & mc & Hin & Hmc & -> & -> & _).
      pose proof (proj1 (forallb_forall _ _) Hall (id, md) Hin) as H.
      cbn beta iota in H. unfold RCHf in Hr. rewrite Hr in H.
      cbn [negb orb] in H.
      apply andb_true_iff in H as [H _].
      apply andb_true_iff in H as [Hp Hn].
      split; [ exact (Hplain_mc id mc Hmc Hp) | ].
      intros He. rewrite He in Hn. discriminate Hn.
    + destruct Hrk as (ids & ->).
      intros id ci Hr Hci'.
      destruct (Hcode id ci Hci') as (md & mc & Hin & Hmc & -> & _ & _).
      pose proof (proj1 (forallb_forall _ _) Hkind (id, md) Hin) as H.
      cbn beta iota in H. unfold RCHf in Hr. rewrite Hr in H.
      cbn [negb orb] in H.
      apply andb_true_iff in H as [Hp _].
      exact (Hplain_mc id mc Hmc Hp).
Qed.

(* A call of [repr_call_r]. *)
Lemma call_rebuilt_r (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (ap : apply_expr)
    (D : list continuation) :
  cinv C en enu ->
  repr_call_r F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr reach C ap
  = true ->
  (∀ z, z ∈ variables_in_apply ap -> fresh z) ->
  ap_exn_continuation ap <> ff -> ap_exn_continuation ap <> ffu ->
  dead_in D (live (Apply ap)) ->
  rr MM CH CSf RCHf D (cc_rebuild en (Apply ap)) (u_rebuild enu (Apply ap)).
Proof.
  intros Hci Hcall Hz Hff Hffu HD. unfold repr_call_r in Hcall.
  apply orb_true_iff in Hcall as [Hcall | Hcall];
    [ exact (call_rebuilt C en enu ap D Hci Hcall Hz Hff Hffu HD) | ].
  destruct (ap_callee ap) as [f | | ] eqn:Hcallee; try discriminate Hcall.
  destruct (code_id_actually_called F sol ap) as [id | ] eqn:Hcalled;
    [ discriminate Hcall | ].
  exact (ind_call_rebuilt C en enu ap f D Hci Hcallee Hcalled Hcall Hz Hff
           Hffu HD).
Qed.

(* ------------------------------------------------------------------------ *)
(* Sets whose representation changes                                        *)
(* ------------------------------------------------------------------------ *)

Lemma lookup_rebuild_value_slots (en : rebuild_env) (bound : list node)
    (vs : list (value_slot * simple)) (w : value_slot) (a : simple) :
  existsb (fun n => s_field_used sol n (Field_value_slot w)) bound = true ->
  lookup w vs = Some a ->
  lookup w (rebuild_value_slots sol en bound vs)
  = Some (rewrite_simple sol en a).
Proof.
  intros Hused. induction vs as [ | [w0 a0] vs IH]; intros Ha;
    [ discriminate Ha | ].
  rewrite lookup_cons in Ha. unfold rebuild_value_slots. cbn [flat_map].
  fold (rebuild_value_slots sol en bound vs).
  destruct (Pos.eqb w w0) eqn:Hw.
  - apply Pos.eqb_eq in Hw. subst w0. injection Ha as <-.
    unfold field_used. rewrite Hused. cbn [app]. apply lookup_head.
  - destruct (existsb (fun n => field_used sol n (Field_value_slot w0))
              bound);
      cbn [app];
      [ rewrite lookup_other by (intros ->; rewrite Pos.eqb_refl in Hw;
                                 discriminate Hw) | ];
      exact (IH Ha).
Qed.

Lemma kept_name_simple (en : rebuild_env) (a : simple) :
  kept_name sol a = true -> rewrite_simple sol en a = a.
Proof.
  destruct a as [z | s | c]; intros H; cbn [rewrite_simple kept_name] in H |- *;
    [ | | reflexivity ].
  - unfold is_var_used. destruct (kind_of_variable en z); try reflexivity;
      rewrite H; reflexivity.
  - unfold is_symbol_used. rewrite H. reflexivity.
Qed.

(* The new slots of a set whose representation changes, as the second
   rebuild makes them, are those of the relation for the slots that the
   first keeps. *)
Lemma changed_slots (en1 en2 : rebuild_env) (x : variable)
    (r : closure_repr) (vs : list (value_slot * simple)) :
  forallb (repr_field_ok F sol unboxed changed x vs) (cr_fields r) = true ->
  (∀ z, z ∈ variables_in_slots vs -> fresh z) ->
  let vs1 := rebuild_value_slots sol en1 (set_closure_nodes F (Node_variable x))
               vs in
  changed_value_slots sol unboxed en2 r vs = new_slots MM r vs1
  ∧ Forall (set_field_ok MM CH vs1) (cr_fields r).
Proof.
  intros Hok Hz vs1. rewrite forallb_forall in Hok.
  assert (Hfield : ∀ fu, fu ∈ cr_fields r ->
            match fu with
            | (Field_value_slot w, u) =>
              new_slot MM vs1 (Field_value_slot w, u)
              = match lookup w vs with
                | Some arg =>
                  match unboxed_of unboxed arg with
                  | Some arg_fields =>
                    match unboxed_field_pairs Value u (Unboxed arg_fields) with
                    | Some pairs =>
                      map (fun '(slot, x, _) => (slot, Var x)) pairs
                    | None => []
                    end
                  | None =>
                    match u with
                    | Not_unboxed slot =>
                      [(slot, rewrite_simple sol_u en2 arg)]
                    | Unboxed _ => []
                    end
                  end
                | None => []
                end
              ∧ set_field_ok MM CH vs1 (Field_value_slot w, u)
            | _ => False
            end).
  { intros [f u] Hin. specialize (Hok _ Hin).
    destruct f as [ | w | | | | | | | ]; try discriminate Hok.
    cbn [repr_field_ok] in Hok. apply andb_true_iff in Hok as [Hused Hok].
    destruct (lookup w vs) as [a | ] eqn:Ha; [ | discriminate Hok ].
    assert (Haz : ∀ z, z ∈ variables_in_simple a -> fresh z).
    { intros z H. apply Hz. unfold variables_in_slots. apply in_flat_map.
      exists (w, a). split; [ exact (lookup_in w a vs Ha) | exact H ]. }
    pose proof (lookup_rebuild_value_slots en1 _ vs w a Hused Ha) as Hl1.
    fold vs1 in Hl1.
    unfold new_slot. rewrite Hl1.
    destruct u as [s | nested].
    - apply andb_true_iff in Hok as [Hpc Hkept].
      rewrite (kept_name_simple en1 a Hkept).
      assert (Hnb : unboxed_of unboxed a = None).
      { apply andb_true_iff in Hpc as [Hpc _].
        destruct a as [z | sym | c]; cbn in Hpc |- *; [ | | reflexivity ];
          unfold FlambdaReaperUnboxingRefinesSpec.plain_variable,
            Keyed.is_bound in Hpc;
          unfold unboxed_of_node;
          destruct (Keyed.lookup _ _ _); [ discriminate Hpc | reflexivity
                                           | discriminate Hpc
                                           | reflexivity ]. }
      rewrite Hnb, (rewrite_c en2 a Hpc Haz), (kept_name_simple en2 a Hkept).
      split; [ reflexivity | ].
      exists a. split; [ rewrite Hl1, (kept_name_simple en1 a Hkept);
                         reflexivity | ].
      exact (plain_c_M' unboxed changed a Hpc Haz).
    - destruct a as [b | | ]; try discriminate Hok.
      destruct (fields_of unboxed b) as [fb | ] eqn:Hb; [ | discriminate Hok ].
      apply andb_true_iff in Hok as [Hok Hpairs].
      apply andb_true_iff in Hok as [Hbu Hbc].
      destruct (unboxed_field_pairs Value (Unboxed nested) (Unboxed fb))
        as [pairs | ] eqn:Hp; [ | discriminate Hpairs ].
      assert (Hbf : fresh b) by (apply Haz; left; reflexivity).
      unfold not_changed, changed_name in Hbc.
      destruct (changed_of_node changed (Node_variable b)) eqn:Hbch;
        [ cbn in Hbc; discriminate Hbc | ].
      assert (HbM : fields_of MM b = Some fb)
        by (rewrite fields_of_M', Hbch; exact Hb).
      rewrite (kept_name_simple en1 (Var b) Hbu).
      cbn [unboxed_of]. change (unboxed_of_node unboxed (Node_variable b))
        with (fields_of unboxed b).
      rewrite Hb, HbM, Hp. split; [ reflexivity | ].
      exists (Var b). split; [ rewrite Hl1, (kept_name_simple en1 (Var b) Hbu);
                               reflexivity | ].
      exists b, fb, pairs. split; [ reflexivity | ].
      split; [ rewrite (CHf_var unboxed changed b Hbf); exact Hbch | ].
      split; [ exact HbM | exact Hp ]. }
  split.
  - unfold changed_value_slots, new_slots. apply flat_map_ext_in'.
    intros [f u] Hin. specialize (Hfield _ Hin).
    destruct f; try contradiction. destruct Hfield as [Heq _].
    rewrite Heq. reflexivity.
  - apply Forall_forall. intros [f u] Hin. specialize (Hfield _ Hin).
    destruct f; try contradiction. exact (proj2 Hfield).
Qed.

Lemma new_slots_keys (vs : list (value_slot * simple))
    (fields : unboxed_fields) :
  Forall (set_field_ok MM CH vs) fields ->
  map fst (flat_map (new_slot MM vs) fields) = map fst (fields_leaves fields).
Proof.
  induction 1 as [ | [f u] fields Hfu _ IH]; [ reflexivity | ].
  cbn [flat_map]. rewrite leaves_cons, !map_app, IH. f_equal.
  destruct f; try contradiction. cbn [set_field_ok] in Hfu.
  destruct Hfu as (a & Ha & Hu). unfold new_slot. rewrite Ha.
  destruct u as [s | nested]; [ reflexivity | ].
  destruct Hu as (b & fb & pairs & -> & _ & Hb & Hp). rewrite Hb, Hp.
  rewrite map_map. cbn [field_kind].
  destruct (pairs_spec _ _ _ _ Hp) as (_ & _ & Hc & _).
  rewrite <- Hc. apply map_ext. intros [[s x] k]. reflexivity.
Qed.

(* ------------------------------------------------------------------------ *)
(* The cases of the rebuilds                                                *)
(* ------------------------------------------------------------------------ *)

Lemma cc_let_cont (en : rebuild_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  cc_rebuild en (Let_cont_non_recursive k h body)
  = let mask := handler_mask sol (handler_is_exn_handler h) h in
    let body' := cc_rebuild (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body')
    then
      Let_cont_non_recursive k
        (rebuild_handler_cc F sol cannot_change code_deps pdc unit_code_ids
           codes ff types en true mask h)
        body'
    else body'.
Proof. reflexivity. Qed.

Lemma u_let_cont (enu : unbox_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  u_rebuild enu (Let_cont_non_recursive k h body)
  = let ds :=
      handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    let body' := u_rebuild (ubind_continuations enu [(k, ds)]) body in
    if among k (free_continuations_of_expr body')
    then
      Let_cont_non_recursive k
        (rebuild_handler_u true F sol cannot_change code_deps pdc
           unit_code_ids codes unboxed ffu changed types enu true ds h)
        body'
    else body'.
Proof. reflexivity. Qed.

Lemma cc_let_cont_rec (en : rebuild_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  cc_rebuild en (Let_cont_recursive handlers body)
  = let en' :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler_cc F sol cannot_change code_deps pdc
                  unit_code_ids codes ff types en' false (handler_mask sol false
                    h)
                  h))
         handlers)
      (cc_rebuild en' body).
Proof. reflexivity. Qed.

Lemma u_let_cont_rec (enu : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  u_rebuild enu (Let_cont_recursive handlers body)
  = let enu' :=
      ubind_continuations enu
        (map (fun '(k, h) => (k, handler_decisions true sol unboxed false h))
           handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler_u true F sol cannot_change code_deps pdc
                  unit_code_ids codes unboxed ffu changed types enu' false
                  (handler_decisions true sol unboxed false h) h))
         handlers)
      (u_rebuild enu' body).
Proof. reflexivity. Qed.

Lemma cc_let_static (en : rebuild_env) (bindings : list static_binding)
    (body : expr) :
  cc_rebuild en (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids
            codes ff types en)
         bindings)
      (cc_rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma u_let_static (enu : unbox_env) (bindings : list static_binding)
    (body : expr) :
  u_rebuild enu (Let_static bindings body)
  = Let_static
      (flat_map
         (rebuild_binding_u true F sol cannot_change code_deps pdc
            unit_code_ids codes unboxed ffu changed types enu)
         bindings)
      (u_rebuild (ubind_variables enu []) body).
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------------ *)
(* Continuations in scope                                                   *)
(* ------------------------------------------------------------------------ *)

Lemma lookup_map_vals {A B : Type} (f : A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun '(k', a) => (k', f a)) l) = option_map f (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ]. cbn [map].
  rewrite !lookup_cons. destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma cinv_conts (C : finite_map (list decision * bool)) (en : rebuild_env)
    (enu : unbox_env) (D : list (continuation * list decision)) :
  cinv C en enu ->
  (∀ k ds, (k, ds) ∈ D -> forallb decided (normed ds) = true) ->
  cinv (map (fun '(k, ds) => (k, (normed ds, true))) D ++ C)
    (bind_continuations en
       (map (fun '(k, ds) => (k, map is_kept_decision ds)) D))
    (ubind_continuations enu D).
Proof.
  intros [Hk Hp Hpu Hd Happ Hout] HD. split.
  - cbn. exact Hk.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros k ds b HC Hb. cbn [ue_decisions ubind_continuations ue_env
                               bind_continuations env_conts].
    rewrite lookup_app in HC. rewrite !lookup_app.
    rewrite (lookup_map_vals (fun ds => (normed ds, true))) in HC.
    rewrite (lookup_map_vals (map is_kept_decision)).
    destruct (lookup k D) as [ds0 | ] eqn:HkD.
    + cbn [option_map] in HC. injection HC as <- <-.
      split; [ exists ds0; split; reflexivity | ].
      split; [ left; cbn [option_map]; rewrite <- is_kept_normed;
               reflexivity | ].
      exact (HD k ds0 (lookup_in k ds0 D HkD)).
    + cbn [option_map]. exact (Happ k ds b HC Hb).
  - intros k HC. cbn [ue_decisions ubind_continuations ue_env
                      bind_continuations env_conts].
    rewrite lookup_app in HC. rewrite !lookup_app.
    rewrite (lookup_map_vals (fun ds => (normed ds, true))) in HC.
    rewrite (lookup_map_vals (map is_kept_decision)).
    destruct (lookup k D); [ discriminate HC | ].
    exact (Hout k HC).
Qed.

(* ------------------------------------------------------------------------ *)
(* The rebuilt terms                                                        *)
(* ------------------------------------------------------------------------ *)

Definition rebuild_ok (e : expr) : Prop :=
  ∀ (en : rebuild_env) (enu : unbox_env)
    (C : finite_map (list decision * bool)) (ctx : option code_id)
    (ce : finite_map (list node)),
    cinv C en enu -> repr (env_kinds en) C e = true -> names_ok e ->
    edges ctx ce e ->
    conts_ok (continuations_of e) ->
    ∀ D : list continuation, dead_in D (live e) ->
    rr MM CH CSf RCHf D (cc_rebuild en e) (u_rebuild enu e).


(* ------------------------------------------------------------------------ *)
(* Code and static bindings                                                 *)
(* ------------------------------------------------------------------------ *)

Lemma rr_bindings_app (l1 l2 l1' l2' : list static_binding) :
  rr_bindings MM CH CSf RCHf l1 l2 -> rr_bindings MM CH CSf RCHf l1' l2' ->
  rr_bindings MM CH CSf RCHf (l1 ++ l1') (l2 ++ l2').
Proof.
  induction 1; intros H'; cbn [app];
    [ exact H' | apply rr_bindings_symbol; auto
    | apply rr_bindings_deleted; auto | eapply rr_bindings_code; eauto ].
Qed.

Lemma params_decided (params : list variable) :
  ∀ pd : list param_decision,
    length params = length pd ->
    map fst
      (flat_map
         (fun '(x, d) => decision_params x (param_decision_at unboxed x d))
         (combine params (map Decided pd)))
    = filter_kept (map is_kept pd) params.
Proof.
  induction params as [ | x params IH]; intros [ | d pd] Hlen;
    cbn in Hlen |- *; try lia; try reflexivity.
  rewrite map_app, IH by lia. destruct d; reflexivity.
Qed.

Lemma changed_metadata_decided (md : code_metadata)
    (pd rd : list param_decision) :
  changed_code_metadata_u md (Decided Delete) (map Decided pd)
    (map Decided rd)
  = changed_code_metadata md pd rd.
Proof.
  unfold changed_code_metadata_u, changed_code_metadata.
  rewrite group_by_parameter_map', arity_of_grouped_decided,
    arity_of_decisions_decided.
  reflexivity.
Qed.

Lemma param_decisions_at_decided (params : list variable) :
  ∀ pd : list param_decision,
    length params = length pd ->
    map (fun '(x, d) => param_decision_at unboxed x d)
      (combine params (map Decided pd))
    = map Decided pd.
Proof.
  induction params as [ | x params IH]; intros [ | d pd] Hlen;
    cbn in Hlen; try lia; [ reflexivity | ].
  cbn [map combine]. rewrite IH by lia. reflexivity.
Qed.

(* With decided decisions, the second rebuild states of the parameters of
   code what the first does. *)
Lemma decisions_subkinds_decided (rw : subkinds_rewrite) (nodes : list node) :
  ∀ (sks : list param_subkind) (dcs : list param_decision),
    decisions_subkinds rw (nodes_with_subkinds nodes sks) (map Decided dcs)
    = filter_kept (map is_kept dcs) (rewritten_at rw nodes sks).
Proof.
  unfold rewritten_at.
  induction nodes as [ | n nodes IH]; intros sks [ | d dcs];
    [ reflexivity | reflexivity | | ].
  - cbn [nodes_with_subkinds map decisions_subkinds filter_kept].
    specialize (IH (tl sks) []). cbn [map] in IH. rewrite IH. reflexivity.
  - cbn [nodes_with_subkinds map decisions_subkinds filter_kept].
    rewrite IH. destruct d; reflexivity.
Qed.

Lemma with_subkinds_cons_c (x : variable) (params : list variable)
    (sks : list param_subkind) :
  with_subkinds (x :: params) sks
  = (x, hd any_value sks) :: with_subkinds params (tl sks).
Proof. destruct sks; reflexivity. Qed.

(* With decided decisions, the second rebuild states of the parameters of
   a handler what the first does. *)
Lemma decide_c_subkinds (rw : subkinds_rewrite)
    (bound : list (variable * flambda_kind)) :
  ∀ (mask : list bool) (sks : list param_subkind),
    length mask = length bound ->
    decisions_subkinds rw
      (nodes_with_subkinds (map (fun '(x, _) => Node_variable x) bound) sks)
      (map decide_c (combine bound mask))
    = filter_kept mask
        (map (fun '(x, ps) => rw (Node_variable x) ps)
           (with_subkinds (map fst bound) sks)).
Proof.
  induction bound as [ | [x kind] bound IH]; intros [ | keep mask] sks Hlen;
    cbn in Hlen; try lia; [ reflexivity | ].
  cbn [map fst combine nodes_with_subkinds decisions_subkinds].
  rewrite with_subkinds_cons_c. cbn [map].
  rewrite IH by lia. destruct keep; reflexivity.
Qed.

Lemma u_rebuild_handler_c (enu : unbox_env) (dead_check : bool)
    (decisions : list decision) (params : list variable)
    (metadata : handler_metadata) (handler : expr) :
  rebuild_handler_u true F sol cannot_change code_deps pdc unit_code_ids
    codes unboxed ffu changed types enu dead_check decisions
    (Handler params metadata handler)
  = let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (map fst (params_of_decisions bound decisions))
      (FlambdaReaperUnboxingRefinesCallsProofs.u_handler_metadata types
         metadata bound decisions)
      (if dead_check && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else u_rebuild (ubind_variables enu bound) handler).
Proof. reflexivity. Qed.

(* The handlers of the fragment state the same in the two rebuilds. *)
Lemma u_handler_metadata_decided (is_exn : bool) (params : list variable)
    (md : handler_metadata) (handler : expr) :
  length params = length (hm_params_kinds md) ->
  forallb decided
    (normed
       (handler_decisions true sol unboxed is_exn
          (Handler params md handler)))
  = true ->
  FlambdaReaperUnboxingRefinesCallsProofs.u_handler_metadata types md
    (with_kinds params (hm_params_kinds md))
    (handler_decisions true sol unboxed is_exn (Handler params md handler))
  = kept_handler_metadata types
      (handler_mask sol is_exn (Handler params md handler)) params md.
Proof.
  intros Hlen Hdec.
  destruct (handler_params_c is_exn params md handler Hlen Hdec)
    as (_ & Hsnd & _).
  assert (Hm : length (handler_mask sol is_exn (Handler params md handler))
               = length (with_kinds params (hm_params_kinds md)))
    by (unfold handler_mask; apply params_to_keep_length).
  assert (Hfst : map fst (with_kinds params (hm_params_kinds md)) = params)
    by (rewrite (with_kinds_combine params _ Hlen);
        apply map_fst_combine; exact Hlen).
  unfold FlambdaReaperUnboxingRefinesCallsProofs.u_handler_metadata,
    kept_handler_metadata.
  rewrite Hsnd.
  destruct types as [t | ]; [ | reflexivity ].
  unfold handler_metadata_with, rewritten_subkinds.
  rewrite decisions_subkinds_normed.
  rewrite (handler_decisions_decided is_exn params md handler Hdec).
  rewrite (decide_c_subkinds (st_rewrite t) _ _ _ Hm), Hfst.
  reflexivity.
Qed.

Lemma changed_code_metadata_with_shape_c (id : code_id)
    (md : code_metadata) (pd rd : list param_decision) :
  cm_params_arity (changed_code_metadata_with types id md pd rd)
  = cm_params_arity (changed_code_metadata md pd rd)
  ∧ cm_result_arity (changed_code_metadata_with types id md pd rd)
    = cm_result_arity (changed_code_metadata md pd rd)
  ∧ cm_is_tupled (changed_code_metadata_with types id md pd rd)
    = cm_is_tupled (changed_code_metadata md pd rd).
Proof.
  unfold changed_code_metadata_with.
  destruct types as [t | ]; [ destruct (lookup id (st_code_deps t)) | ];
    repeat split.
Qed.

(* The exception continuation of code that may be raised to: the second
   rebuild keeps the exception, with a use or not. *)
Lemma code_exn_ds_cases (id : code_id) (cd : code_dep) (ds : list decision) :
  lookup id code_deps = Some cd ->
  code_exn_ds sol code_deps unboxed id = ds -> ds <> [] ->
  normed (fixed_arity_decisions sol unboxed [cd_exn cd] [Value]) = ds
  ∧ forallb keeps ds = true ∧ forallb decided ds = true.
Proof.
  intros Hcd Hds Hne. unfold code_exn_ds, code_exn_decision in Hds.
  rewrite Hcd in Hds. unfold fixed_arity_decisions in *.
  cbn [combine map] in *.
  destruct (unboxed_of_node unboxed (cd_exn cd)) as [fields | ];
    [ subst ds; contradiction Hne; reflexivity | ].
  destruct (is_node_used sol (cd_exn cd) Value); subst ds;
    repeat split.
Qed.

(* The return continuation of code that keeps its calling convention:
   every result kept, with a use or not. *)
Lemma fixed_keeps (nodes : list node) (kinds : list flambda_kind) :
  forallb decided (normed (fixed_arity_decisions sol unboxed nodes kinds))
  = true ->
  forallb keeps (normed (fixed_arity_decisions sol unboxed nodes kinds))
  = true.
Proof.
  unfold normed, fixed_arity_decisions. rewrite !map_map.
  intros H. rewrite forallb_forall in H |- *. intros d Hd.
  specialize (H d Hd). apply in_map_iff in Hd as ([n k] & <- & _).
  destruct (unboxed_of_node unboxed n); [ discriminate H | ].
  destruct (is_node_used sol n k); reflexivity.
Qed.

(* Code of the fragment is rebuilt into related code. *)
Lemma code_rebuilt (id : code_id) (c : code) :
  rebuild_ok (code_body c) ->
  repr_code_r F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr reach id c
  = true ->
  (∀ z, z ∈ variables_in_code c -> fresh z) ->
  conts_ok (continuations_in_code c) ->
  FlambdaReaperFlowProofs.code_edges (graph_of_facts F) noted deps cds
    kept_slots codes0 symbols True False id c ->
  ∃ ci : code_info,
    code_frame MM CH CSf id
      (rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids codes
         ff types id c)
      (rebuild_code_u true F sol cannot_change code_deps pdc
         unit_code_ids codes unboxed ffu changed types id c)
      ci
    ∧ rr MM CH CSf RCHf
        (code_dead ci
           (rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids
              codes ff types id c))
        (code_body
           (rebuild_code_cc F sol cannot_change code_deps pdc unit_code_ids
              codes ff types id c))
        (code_body
           (rebuild_code_u true F sol cannot_change code_deps pdc
              unit_code_ids codes unboxed ffu changed types id c)).
Proof.
  intros IH Hrepr Hnames Hconts Hedges.
  destruct c as [params mc ret exn md body].
  cbn [repr_code_r] in Hrepr.
  apply andb_true_iff in Hrepr as [Hrepr Hlive].
  apply andb_true_iff in Hrepr as [Hrepr Hchange].
  apply andb_true_iff in Hrepr as [Hrepr Hre].
  apply andb_true_iff in Hrepr as [Hrepr Hmcu].
  apply andb_true_iff in Hrepr as [Hrepr Hps].
  apply andb_true_iff in Hrepr as [Hmdeq Hmceq].
  destruct (lookup id codes) as [md' | ] eqn:Hmd; [ | discriminate Hmdeq ].
  destruct (lookup id my_closures) as [mc' | ] eqn:Hmc;
    [ | discriminate Hmceq ].
  apply code_metadata_eqb_eq in Hmdeq. subst md'.
  apply Pos.eqb_eq in Hmceq. subst mc'.
  apply negb_true_iff, Pos.eqb_neq in Hre.
  cbn [variables_in_code] in Hnames.
  assert (Hmcf : fresh mc) by (apply Hnames; left; reflexivity).
  assert (Hpf : ∀ x, x ∈ params -> fresh x)
    by (intros x Hx; apply Hnames; right; apply in_or_app; left; exact Hx).
  set (ci := Code_info (CH mc) (length (cm_params_arity md)) (nr id)).
  assert (HCS : CSf id = Some ci)
    by (unfold CSf; rewrite Hmd, Hmc; reflexivity).
  assert (Hmcok : my_closure_ok MM CH ci mc).
  { unfold my_closure_ok. cbn [ci_my_closure ci].
    destruct (CH mc) as [r | ] eqn:Hr; [ reflexivity | ].
    rewrite (CHf_var unboxed changed mc Hmcf) in Hr.
    rewrite plain_variable_M', Hr, andb_true_r.
    unfold plain_variable. rewrite Hmcf. unfold Keyed.is_bound.
    destruct (fields_of unboxed mc) eqn:E; [ discriminate Hmcu | ].
    unfold fields_of in E. rewrite E. reflexivity. }
  assert (Hpsplain : forallb (plain_variable MM) params = true).
  { apply forallb_forall. intros x Hx.
    apply (plain_c_var_M' unboxed changed x);
      [ exact (proj1 (forallb_forall _ params) Hps x Hx) | exact (Hpf x Hx) ].
    }
  assert (Hdead : dead_in (if nr id then [ret] else [])
                    (live_conts F sol code_deps codes body)).
  { destruct (nr id) eqn:Hnrid; [ | intros k [] ].
    intros k [Hk0 | []] Hk. subst k.
    apply among_In in Hk. rewrite Hk in Hlive. discriminate Hlive. }
  destruct Hedges as (cd' & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _
                      & Hbedges).
  assert (Hbn : ∀ z, z ∈ variables_of body -> fresh z)
    by (intros z Hz; apply Hnames; right; apply in_or_app; right; exact Hz).
  assert (Hbc : conts_ok (continuations_of body))
    by (intros k Hk; apply Hconts; right; right; exact Hk).
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md)
    as [ | pd rd] eqn:Hch.
  - (* Code that keeps its calling convention: the same parameters, its
       return continuation with no decision in the first rebuild and those
       of a continuation of fixed arity in the second, each keeping. *)
    apply andb_true_iff in Hchange as [Htup Hcd].
    apply negb_true_iff in Htup.
    destruct (lookup id code_deps) as [cd | ] eqn:Hcdid;
      [ | discriminate Hcd ].
    apply andb_true_iff in Hcd as [Hrds Hbody].
    assert (Hncc : code_calling_convention_change sol cannot_change
                     code_deps unit_code_ids id md
                   = Not_changing_calling_convention).
    { unfold code_change_u in Hch. unfold code_calling_convention_change.
      destruct (cannot_change_calling_convention cannot_change unit_code_ids
                  id);
        [ reflexivity | ].
      rewrite Hcdid in Hch. discriminate Hch. }
    assert (Hka : cm_params_arity (kept_code_metadata types id md)
                  = cm_params_arity md
                  ∧ cm_is_tupled (kept_code_metadata types id md)
                    = cm_is_tupled md).
    { unfold kept_code_metadata.
      destruct types as [t | ]; [ destruct (lookup id (st_code_deps t)) | ];
        split; reflexivity. }
    destruct Hka as [Hka Hkt].
    cbn [rebuild_code_cc rebuild_code_u]. rewrite Hncc, Hch.
    unfold code_continuation_decisions. rewrite Hcdid.
    exists ci. split.
    + split; [ exact HCS | ].
      cbn [code_params code_my_closure code_return_continuation
           code_exn_continuation code_params_arity code_result_arity
           code_is_tupled code_params_subkinds].
      rewrite Hka, Hkt.
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ exact Htup | split; [ exact Htup | ] ].
      split; [ reflexivity | ].
      split; [ exact Hpsplain | ].
      split; [ exact Hmcok | reflexivity ].
    + cbn [code_body].
      destruct (existsb (fun '(x, kind) => is_dead_var sol x kind) _);
        [ constructor | ].
      apply (IH _ _
               [(ret,
                 (normed
                    (fixed_arity_decisions sol unboxed (cd_return cd)
                       (unarize (cm_result_arity md))),
                  true));
                (exn, (code_exn_ds sol code_deps unboxed id, false))]
               (Some id)
               [(exn, [cd_exn cd']); (ret, cd_return cd')]).
      * split.
        -- reflexivity.
        -- reflexivity.
        -- reflexivity.
        -- reflexivity.
        -- intros k ds b HC Hb.
           cbn [ue_decisions ubind_variables ubind_variables_then ubind_set
                ubind_continuations empty_unbox_env app env_conts
                bind_variables bind_variables_then bind_continuations
                empty_env].
           rewrite !lookup_cons in HC. rewrite !lookup_cons.
           destruct (Pos.eqb k ret) eqn:Hk.
           ++ injection HC as <- <-.
              split; [ eexists; split; reflexivity | ].
              split; [ right; split; [ reflexivity | ] | exact Hrds ].
              exact (fixed_keeps _ _ Hrds).
           ++ destruct (Pos.eqb k exn) eqn:Hke; [ | discriminate HC ].
              injection HC as <- <-.
              destruct Hb as [Hb | Hb]; [ discriminate Hb | ].
              destruct (code_exn_ds_cases id cd _ Hcdid eq_refl Hb)
                as (Hn & Hkeeps & Hdec).
              split; [ eexists; split; [ reflexivity | exact Hn ] | ].
              split; [ right; split; [ reflexivity | exact Hkeeps ]
                     | exact Hdec ].
        -- intros k HC.
           cbn [ue_decisions ubind_variables ubind_variables_then ubind_set
                ubind_continuations empty_unbox_env app env_conts
                bind_variables bind_variables_then bind_continuations
                empty_env].
           rewrite !lookup_cons in HC. rewrite !lookup_cons.
           destruct (Pos.eqb k ret); [ discriminate HC | ].
           destruct (Pos.eqb k exn); [ discriminate HC | ].
           split; reflexivity.
      * cbn [env_kinds bind_variables bind_variables_then bind_continuations
             empty_env].
        rewrite app_nil_r. exact Hbody.
      * exact Hbn.
      * exact Hbedges.
      * exact Hbc.
      * exact Hdead.
  - (* Code that changes its calling convention. *)
    apply andb_true_iff in Hchange as [Hchange Hbody].
    apply andb_true_iff in Hchange as [Hchange Hmcd].
    apply andb_true_iff in Hchange as [Hchange Hrlen].
    apply andb_true_iff in Hchange as [Hchange Hplen].
    apply andb_true_iff in Hchange as [Hchange Hlen].
    apply andb_true_iff in Hchange as [Hpd Hrd].
    destruct (code_change_decided id md pd rd Hch Hpd Hrd)
      as (cd & pd_cc & rd_cc & Hcd & Hcc & -> & ->).
    apply Nat.eqb_eq in Hlen. rewrite length_map in Hlen.
    cbn [rebuild_code_cc rebuild_code_u]. rewrite Hcc, Hch.
    unfold code_continuation_decisions. rewrite Hcd.
    destruct (my_closure_decision_u cannot_change code_deps unit_code_ids
                unboxed id);
      [ discriminate Hmcd | ].
    cbn [decision_params app]. rewrite (params_decided params pd_cc Hlen).
    rewrite changed_metadata_decided.
    destruct (changed_code_metadata_with_shape_c id md pd_cc rd_cc)
      as (Hca & Hcr & Hct).
    exists ci. split.
    + match goal with
      | |- code_frame _ _ _ _ (Code _ _ _ _ ?MA _) (Code _ _ _ _ ?MB _) _ =>
        replace MB with MA
      end.
      2: { unfold changed_code_metadata_with.
           destruct types as [t | ]; [ | reflexivity ].
           destruct (lookup id (st_code_deps t)) as [cd'' | ];
             [ | reflexivity ].
           cbn [app]. rewrite (param_decisions_at_decided params pd_cc Hlen).
           rewrite !decisions_subkinds_decided. reflexivity. }
      split; [ exact HCS | ].
      cbn [code_params code_my_closure code_return_continuation
           code_exn_continuation code_params_arity code_result_arity
           code_is_tupled].
      rewrite Hca, Hcr, Hct.
      cbn [changed_code_metadata cm_params_arity cm_is_tupled].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split.
      { cbn [ci_nparams ci]. unfold arity_of_grouped_decisions.
        rewrite length_map, group_by_parameter_length'. reflexivity. }
      split; [ apply filter_kept_forallb; exact Hpsplain | ].
      split; [ exact Hmcok | reflexivity ].
    + cbn [code_body].
      destruct (existsb (fun '(x, kind) => is_dead_var sol x kind) _);
        [ constructor | ].
      apply (IH _ _
               [(ret, (map Decided rd_cc, true));
                (exn, (code_exn_ds sol code_deps unboxed id, false))]
               (Some id)
               [(exn, [cd_exn cd']); (ret, cd_return cd')]).
      * split.
        -- reflexivity.
        -- reflexivity.
        -- reflexivity.
        -- reflexivity.
        -- intros k ds b HC Hb.
           cbn [ue_decisions ubind_variables ubind_variables_then ubind_set
                ubind_continuations empty_unbox_env app env_conts
                bind_variables bind_variables_then bind_continuations
                empty_env].
           rewrite !lookup_cons in HC. rewrite !lookup_cons.
           destruct (Pos.eqb k ret) eqn:Hk.
           ++ injection HC as <- <-.
              split; [ exists (map Decided rd_cc);
                       split; [ reflexivity | exact (normed_decided _ Hrd) ]
                     | ].
              split; [ left; rewrite map_map; reflexivity | exact Hrd ].
           ++ destruct (Pos.eqb k exn) eqn:Hke; [ | discriminate HC ].
              injection HC as <- <-.
              destruct Hb as [Hb | Hb]; [ discriminate Hb | ].
              destruct (code_exn_ds_cases id cd _ Hcd eq_refl Hb)
                as (Hn & Hkeeps & Hdec).
              split; [ eexists; split; [ reflexivity | exact Hn ] | ].
              split; [ right; split; [ reflexivity | exact Hkeeps ]
                     | exact Hdec ].
        -- intros k HC.
           cbn [ue_decisions ubind_variables ubind_variables_then ubind_set
                ubind_continuations empty_unbox_env app env_conts
                bind_variables bind_variables_then bind_continuations
                empty_env].
           rewrite !lookup_cons in HC. rewrite !lookup_cons.
           destruct (Pos.eqb k ret); [ discriminate HC | ].
           destruct (Pos.eqb k exn); [ discriminate HC | ].
           split; reflexivity.
      * cbn [env_kinds bind_variables bind_variables_then bind_continuations
             empty_env].
        rewrite app_nil_r. exact Hbody.
      * exact Hbn.
      * exact Hbedges.
      * exact Hbc.
      * exact Hdead.
Qed.


Lemma bindings_rebuilt (C : finite_map (list decision * bool))
    (en : rebuild_env) (enu : unbox_env) (bindings : list static_binding) :
  (∀ id c, Bind_code id c ∈ bindings -> rebuild_ok (code_body c)) ->
  cinv C en enu ->
  forallb
    (repr_binding_r F sol cannot_change code_deps unit_code_ids codes unboxed
       changed my_closures nr reach)
    bindings
  = true ->
  (∀ z, z ∈ flat_map variables_in_binding bindings -> fresh z) ->
  conts_ok (flat_map continuations_in_binding bindings) ->
  every
    (FlambdaReaperFlowProofs.binding_edges (graph_of_facts F) noted deps cds
       kept_slots codes0 symbols True False)
    bindings ->
  rr_bindings MM CH CSf RCHf
    (flat_map
       (rebuild_binding_cc F sol cannot_change code_deps pdc unit_code_ids
          codes ff types en)
       bindings)
    (flat_map
       (rebuild_binding_u true F sol cannot_change code_deps pdc
          unit_code_ids codes unboxed ffu changed types enu)
       bindings).
Proof.
  intros IHcode Hci. induction bindings as [ | b bs IH];
    intros Hrepr Hnames Hconts Hedges; [ constructor | ].
  cbn [forallb] in Hrepr. apply andb_true_iff in Hrepr as [Hb Hbs].
  destruct Hedges as [Hbe Hbse].
  cbn [flat_map] in Hnames |- *. apply rr_bindings_app.
  2: { apply IH; [ | exact Hbs | | | exact Hbse ].
       - intros id c Hin. exact (IHcode id c (or_intror Hin)).
       - intros z Hz. apply Hnames, in_or_app. right. exact Hz.
       - eapply conts_ok_incl; [ | exact Hconts ]. intros k Hk.
         cbn [flat_map]. apply in_or_app. right. exact Hk. }
  assert (Hbn : ∀ z, z ∈ variables_in_binding b -> fresh z)
    by (intros z Hz; apply Hnames, in_or_app; left; exact Hz).
  destruct b as [id c | id | s const].
  - cbn [rebuild_binding_cc rebuild_binding_u].
    destruct (is_code_id_used sol id).
    + destruct (code_rebuilt id c (IHcode id c (or_introl eq_refl)) Hb Hbn
                  (conts_ok_incl _ _
                     (fun k Hk => in_or_app _ _ k (or_introl Hk)) Hconts)
                  (proj2 Hbe))
        as (ci & Hframe & Hbody).
      eapply rr_bindings_code; [ exact Hframe | exact Hbody | constructor ].
    + apply rr_bindings_deleted. constructor.
  - cbn [rebuild_binding_cc rebuild_binding_u].
    apply rr_bindings_deleted. constructor.
  - destruct const as [ | | tag fields | | | ]; cbn [repr_binding_r] in Hb;
      try discriminate Hb.
    cbn [rebuild_binding_cc rebuild_binding_u static_set_repr].
    destruct (static_const_is_used sol s (Static_const_block tag fields));
      [ | constructor ].
    cbn [rebuild_static_const]. cbn [variables_in_binding
                                     variables_in_static_const] in Hbn.
    rewrite (rebuild_fields_u sol unboxed (ue_env enu) _ fields
               (plain_cs_strong fields Hb Hbn) 0).
    rewrite (rebuild_fields_kinds sol (ue_env enu) en _ fields
               (ci_kinds _ _ _ Hci) 0).
    apply rr_bindings_symbol; [ | constructor ].
    cbn [local_binding].
    exact (rebuild_fields_plain sol MM en _ fields (plain_cs_M' fields Hb Hbn)
             0).
Qed.


(* ------------------------------------------------------------------------ *)
(* The [Let]s of the fragment                                               *)
(* ------------------------------------------------------------------------ *)

Lemma plain_MM (s : simple) :
  plain_simple unboxed s = true -> not_changed changed s = true ->
  plain_simple MM s = true.
Proof.
  intros Hs Hc. destruct s as [x | sym | c]; cbn [plain_simple] in *;
    [ | | reflexivity ].
  - rewrite plain_variable_M', Hs. exact Hc.
  - unfold Keyed.is_bound in *. rewrite lookup_M'.
    unfold not_changed, changed_name, changed_of_node in Hc.
    destruct (Keyed.lookup node_eqb (Node_symbol sym) changed);
      [ discriminate Hc | exact Hs ].
Qed.

Lemma plains_MM (l : list simple) :
  forallb (plain_simple unboxed) l = true ->
  forallb (not_changed changed) l = true ->
  forallb (plain_simple MM) l = true.
Proof.
  induction l as [ | s l IH]; intros H1 H2; [ reflexivity | ].
  cbn [forallb] in *. apply andb_true_iff in H1 as [H1 H1'].
  apply andb_true_iff in H2 as [H2 H2'].
  rewrite (plain_MM s H1 H2). exact (IH H1' H2').
Qed.

Lemma named_plain_MM (d : named) :
  FlambdaReaperUnboxingRefinesProofs.named_plain unboxed d = true ->
  no_changed changed d = true ->
  FlambdaReaperUnboxingRefinesProofs.named_plain MM d = true.
Proof.
  intros Hd Hc. unfold no_changed in Hc.
  destruct d as [s | p args | | | | | ]; cbn in Hd, Hc |- *;
    try discriminate Hd.
  - rewrite andb_true_r in Hc. exact (plain_MM s Hd Hc).
  - exact (plains_MM args Hd Hc).
Qed.

Lemma not_changed_rewrite (en : rebuild_env) (s : simple) :
  not_changed changed s = true ->
  not_changed changed (rewrite_simple sol en s) = true.
Proof.
  destruct s as [x | sym | c]; cbn [rewrite_simple]; intros H; [ | | exact H ].
  - destruct (is_var_used sol x _); [ exact H | reflexivity ].
  - destruct (is_symbol_used sol sym); [ exact H | reflexivity ].
Qed.

Lemma not_changed_fields (en : rebuild_env) (base : node) (args : list simple) :
  forallb (not_changed changed) args = true ->
  ∀ i : nat, forallb (not_changed changed) (rebuild_fields sol en base i args)
             = true.
Proof.
  induction args as [ | s args IH]; intros H i; [ reflexivity | ].
  cbn [forallb] in H. apply andb_true_iff in H as [Hs H].
  cbn [rebuild_fields forallb]. rewrite (IH H).
  destruct (field_used sol base _); [ | reflexivity ].
  rewrite (not_changed_rewrite en s Hs). reflexivity.
Qed.

(* The read of a [Let] that the rebuild keeps of a plain variable, if no
   name it reads changes. *)
Lemma rebuild_let_u_plain_c (en : rebuild_env) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  FlambdaReaperUnboxingRefinesProofs.named_plain unboxed d = true ->
  no_changed changed d = true ->
  rebuild_let_u sol unboxed changed en x kind d body
  = Let x kind (rebuild_named sol en x d) body.
Proof.
  intros Hd Hc. unfold rebuild_let_u.
  destruct (field_read d) as [[f arg] | ] eqn:Hread.
  - assert (Harg : plain_simple unboxed arg = true
                   ∧ not_changed changed arg = true).
    { destruct d as [s0 | p args | | | | | ]; simpl in Hd; try discriminate Hd;
        [ discriminate Hread | ].
      unfold no_changed in Hc. cbn [named_simples] in Hc.
      destruct p as [p | p | p | p | p]; simpl in Hread; try discriminate Hread.
      destruct p; simpl in Hread; try discriminate Hread;
        destruct args as [ | a [ | ]]; simpl in Hread; try discriminate Hread;
        try (destruct variant_only; simpl in Hread; try discriminate Hread);
        injection Hread as _ <-; simpl in Hd, Hc; rewrite andb_true_r in Hd, Hc;
        split; assumption. }
    destruct Harg as [Harg Hch].
    rewrite (plain_not_bound unboxed arg Harg).
    assert (Hno : changed_of changed arg = None).
    { destruct arg as [z | sym | c]; cbn in Hch |- *; [ | | reflexivity ];
        unfold changed_name in Hch;
        destruct (changed_of_node changed _); [ discriminate Hch | reflexivity
                                               | discriminate Hch
                                               | reflexivity ]. }
    rewrite Hno, (named_plain_u sol unboxed en x d Hd). reflexivity.
  - rewrite (named_plain_u sol unboxed en x d Hd). reflexivity.
Qed.

Lemma alloc_field_simples (d : named) (f : field) (kind : flambda_kind)
    (s : simple) :
  alloc_field d f kind s -> s ∈ named_simples d.
Proof.
  intros H. destruct d as [ | p args | | | | | ]; try contradiction.
  destruct p as [p | p | p | p | p]; try contradiction.
  - destruct p; try contradiction. destruct args as [ | c [ | ]];
      try contradiction.
    destruct H as (_ & _ & ->). left. reflexivity.
  - destruct p as [[tag] | ]; try contradiction.
    destruct H as (i & _ & _ & Hnth). exact (nth_error_In _ _ Hnth).
Qed.

(* The entries of an allocation unboxed, of names that do not change, are
   those of the relation. *)
Lemma alloc_entries_MM (x : variable) (d : named) (fields : unboxed_fields)
    (ls : list (list (variable * flambda_kind * simple))) :
  forallb (not_changed changed) (named_simples d) = true ->
  Forall2 (alloc_entry unboxed x d) fields ls ->
  Forall2 (alloc_entry MM x d) fields ls.
Proof.
  intros Hd. induction 1 as [ | [f u] l fields ls Hent _ IH];
    constructor; [ | exact IH ].
  rewrite forallb_forall in Hd.
  destruct u as [v | nested]; cbn [alloc_entry] in Hent |- *.
  - destruct Hent as (kind & s & -> & Hf & Hs). exists kind, s.
    split; [ reflexivity | split; [ exact Hf | ] ].
    exact (plain_MM s Hs (Hd s (alloc_field_simples d f kind s Hf))).
  - destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hp & ->).
    exists b, fb, pairs. split; [ exact Hf | ].
    split; [ | split; [ exact Hxb | split; [ exact Hp | reflexivity ] ] ].
    pose proof (Hd _ (alloc_field_simples d f Value (Var b) Hf)) as Hbc.
    cbn [not_changed] in Hbc. unfold changed_name in Hbc.
    rewrite fields_of_M'.
    destruct (changed_of_node changed (Node_variable b));
      [ discriminate Hbc | exact Hb ].
Qed.

Lemma rebuild_value_slots_c (en1 en2 : rebuild_env) (bound : list node)
    (vs : list (value_slot * simple)) :
  env_kinds en1 = env_kinds en2 ->
  forallb (fun ws => plain_c unboxed changed (snd ws)) vs = true ->
  (∀ z, z ∈ variables_in_slots vs -> fresh z) ->
  rebuild_value_slots sol_u en2 bound vs = rebuild_value_slots sol en1 bound vs
  ∧ forallb (fun ws => plain_simple MM (snd ws))
      (rebuild_value_slots sol en1 bound vs)
    = true.
Proof.
  intros Hk. induction vs as [ | [w s] vs IH]; intros Hvs Hz;
    [ split; reflexivity | ].
  cbn [forallb snd] in Hvs. apply andb_true_iff in Hvs as [Hs Hvs].
  assert (Hsz : ∀ z, z ∈ variables_in_simple s -> fresh z)
    by (intros z H; apply Hz; unfold variables_in_slots; cbn [flat_map snd];
        apply in_or_app; left; exact H).
  destruct IH as [IH1 IH2];
    [ exact Hvs
    | intros z H; apply Hz; unfold variables_in_slots; cbn [flat_map snd];
      apply in_or_app; right; exact H | ].
  unfold rebuild_value_slots in *. cbn [flat_map].
  rewrite IH1. unfold field_used.
  change (s_field_used sol_u) with (s_field_used sol).
  destruct (existsb _ bound); cbn [app forallb snd];
    [ | split; [ reflexivity | exact IH2 ] ].
  rewrite (rewrite_c en2 s Hs Hsz),
    (rewrite_simple_kinds sol en2 en1 s (eq_sym Hk)).
  split; [ reflexivity | ].
  rewrite (rewrite_plain_M' en1 s (plain_c_M' unboxed changed s Hs Hsz)).
  exact IH2.
Qed.



Lemma read_block_like (p : unary_primitive) (y : variable) (f : field) :
  read_of (Prim (Unary p) [Var y]) = Some (f, Var y) ->
  is_block_like_field f = true.
Proof.
  intros H. destruct p; try discriminate H; simpl in H;
    injection H as <-; reflexivity.
Qed.

Lemma field_read_slot (d : named) (w : value_slot) (s : simple) :
  field_read d = Some (Field_value_slot w, s) ->
  d = Project_value_slot w s ∨ ∃ fs : function_slot,
                                d = Project_value_slot_from fs w s.
Proof.
  intros H.
  destruct d as [s0 | p args | id vs | slot decls vs | w0 s0 | fs w0 s0
                | mf mt s0];
    cbn [field_read] in H; try discriminate H.
  - destruct p as [p | p | p | p | p]; try discriminate H.
    destruct p; try discriminate H; destruct args as [ | a [ | ]];
      try discriminate H;
      try (destruct variant_only; discriminate H).
  - injection H as -> ->. left. reflexivity.
  - injection H as -> ->. right. exists fs. reflexivity.
Qed.

Lemma let_rebuilt (C : finite_map (list decision * bool)) (en : rebuild_env)
    (enu : unbox_env) (ctx : option code_id) (ce : finite_map (list node))
    (x : variable) (kind : flambda_kind) (d : named) (body : expr) :
  cinv C en enu ->
  repr (env_kinds en) C (Let x kind d body) = true ->
  names_ok (Let x kind d body) ->
  edges ctx ce (Let x kind d body) ->
  ∀ D : list continuation,
  rr MM CH CSf RCHf D (cc_rebuild (bind_variables en [(x, kind)]) body)
    (u_rebuild (ubind_variables enu [(x, kind)]) body) ->
  rr MM CH CSf RCHf D (cc_rebuild en (Let x kind d body))
    (u_rebuild enu (Let x kind d body)).
Proof.
  intros Hci Hrepr Hnames Hedges D IHrr.
  cbn [repr_expr_r] in Hrepr.
  apply andb_true_iff in Hrepr as [Hrepr _].
  apply andb_true_iff in Hrepr as [Hkind Hbind].
  destruct Hedges as [Hnamed_edges _].
  unfold names_ok in Hnames. cbn [variables_of] in Hnames.
  assert (Hxf : fresh x) by (apply Hnames; left; reflexivity).
  assert (Hdf : ∀ z, z ∈ variables_in_named d -> fresh z)
    by (intros z Hz; apply Hnames; right; apply in_or_app; left; exact Hz).
  unfold cc_rebuild, u_rebuild in *. cbn [rebuild_expr_cc rebuild_expr_u].
  unfold is_pending_projection.
  rewrite (ci_proj _ _ _ Hci), (ci_proj_u _ _ _ Hci).
  rewrite (rebuild_named_kinds sol en (ue_env enu) x d
             (eq_sym (ci_kinds _ _ _ Hci))).
  assert (Hkd : ∀ s, simple_is_dead sol (ue_env enu) s
                     = simple_is_dead sol en s)
    by (intros s; destruct s; cbn; try reflexivity;
        unfold kind_of_variable; rewrite (ci_kinds _ _ _ Hci); reflexivity).
  unfold changed_name in Hbind.
  change (unboxed_of_node unboxed (Node_variable x)) with (fields_of unboxed x).
  assert (Hun : is_unboxed_name unboxed (Node_variable x)
                = negb (is_none (fields_of unboxed x)))
    by (unfold is_unboxed_name, Keyed.is_bound, fields_of;
        destruct (Keyed.lookup _ _ _); reflexivity).
  destruct (fields_of unboxed x) as [fields | ] eqn:Hx;
    destruct (changed_of_node changed (Node_variable x)) as [r | ] eqn:Hxc;
    try discriminate Hbind.
  - (* An unboxed variable. *)
    assert (HxM : fields_of MM x = Some fields)
      by (rewrite fields_of_M', Hxc; exact Hx).
    assert (HxCH : CH x = None)
      by (rewrite (CHf_var unboxed changed x Hxf); exact Hxc).
    assert (HxF : among x (field_variables MM) = false)
      by (rewrite field_variables_M'; exact Hxf).
    apply orb_true_iff in Hbind as [Hb | Hslot].
    + apply andb_true_iff in Hb as [Hub Hnc].
      apply (strong_unboxed_binding unboxed sol _ x kind d fields Hub) in Hdf.
      rename Hdf into Hub'.
      destruct (unboxed_binding_cases sol unboxed _ x kind d fields Hub')
        as (-> & [(tag & args & -> & Hfields)
                 | [(bn & c & -> & Hc & Hfields)
                   | [(y & fields_y & -> & Hne & Hy & Hflat & Hyused)
                     | (bk & mut & i & y & fy & nested & -> & Hne & Hy & Hny
                        & Halike)]]]);
        cbn beta iota;
        (destruct (is_dead_var sol x Value) eqn:Hdead;
         [ apply rr_invalid | ]);
        (destruct (is_var_used sol x Value) eqn:Hused;
         [ | exact IHrr ]);
        cbn iota.
      * (* A block. *)
        cbn [rebuild_named]. simpl in Hnamed_edges.
        destruct Hnamed_edges as [_ Hnamed_edges].
        destruct (bind_block_fields_lets2 sol unboxed (ue_env enu) x tag args
                    fields)
          with (body := rebuild_expr_u true F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu changed types
                          (ubind_variables enu [(x, Value)]) body)
          as (ls & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & a & -> & Hnth & Hkind' & Hcase).
          rewrite kinds_env_kind in Hkind'.
          destruct (field_kept F sol unboxed is_local sol_closed local_slots
                      fields_read x fields _ u a (ue_env enu) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, a. repeat split; try assumption.
          destruct a; cbn [simple_kind] in Hkind' |- *;
            [ | reflexivity | exact Hkind' ].
          unfold kind_of_variable in Hkind' |- *.
          rewrite (ci_kinds _ _ _ Hci). exact Hkind'. }
        unfold rebuild_unboxed_binding. rewrite Hl.
        apply (rr_alloc MM CH CSf RCHf D x _ fields ls); try assumption.
        -- apply alloc_entries_MM; [ | exact Hentries ].
           cbn [named_simples]. apply not_changed_fields. exact Hnc.
        -- intros q0 args0 Hq0; discriminate Hq0.
      * (* A boxed number. *)
        simpl in Hnamed_edges. destruct Hnamed_edges as [_ Hnamed_edges].
        unfold no_changed in Hnc. cbn [named_simples forallb] in Hnc.
        rewrite andb_true_r in Hnc.
        destruct (bind_boxed_number_lets unboxed x bn c fields
                    (rebuild_named sol (ue_env enu) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := rebuild_expr_u true F sol cannot_change code_deps
                          pdc unit_code_ids codes unboxed ffu changed types
                          (ubind_variables enu [(x, Value)]) body)
          as (ls & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept F sol unboxed is_local sol_closed local_slots
                      fields_read x _ _ (Not_unboxed v) c (ue_env enu) Hx
                      (or_introl eq_refl) (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named]. unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite sol _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. rewrite Hl.
        apply (rr_alloc MM CH CSf RCHf D x _ fields ls); try assumption.
        -- apply alloc_entries_MM; [ | exact Hentries ].
           cbn [rebuild_named]. destruct (field_used sol _ _); cbn;
             [ rewrite (not_changed_rewrite _ c Hnc) | ]; reflexivity.
        -- intros q0 args0 Hq0; cbn [rebuild_named] in Hq0;
             destruct (field_used _ _ _); discriminate Hq0.
      * (* An alias of an unboxed variable. *)
        cbn [rebuild_named rewrite_simple].
        unfold kind_of_variable in Hyused |- *.
        cbn [env_kinds kinds_env] in Hyused.
        rewrite (ci_kinds _ _ _ Hci), Hyused.
        unfold rebuild_unboxed_binding.
        destruct (bind_fields_pairs fields fields_y
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu changed types
                       (ubind_variables enu [(x, Value)]) body)
                    Hflat)
          as (pairs & Hentries & Hl).
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hl.
        apply (rr_alias MM CH CSf RCHf D x y fields fields_y pairs);
          try assumption.
        unfold no_changed in Hnc.
        cbn [named_simples forallb not_changed] in Hnc.
        unfold changed_name in Hnc. rewrite fields_of_M'.
        destruct (changed_of_node changed (Node_variable y));
          [ cbn in Hnc; discriminate Hnc | exact Hy ].
      * (* The load of a field unboxed in turn. *)
        cbn [rebuild_named]. unfold rebuild_unboxed_binding, load_unboxed_field.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hny.
        destruct (bind_fields_pairs fields nested
                    (rebuild_expr_u true F sol cannot_change code_deps pdc
                       unit_code_ids codes unboxed ffu changed types
                       (ubind_variables enu [(x, Value)]) body)
                    Halike)
          as (pairs & Hentries & Hl).
        rewrite Hl.
        apply (rr_load MM CH CSf RCHf D x _ (Field_block (Z.to_nat i) Value)
                 (rewrite_simple sol (ue_env enu) (Var y)) y fields fy nested
                 pairs); try assumption.
        -- reflexivity.
        -- simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
           right. eexists. reflexivity.
        -- unfold no_changed in Hnc.
           cbn [named_simples forallb not_changed] in Hnc.
           unfold changed_name in Hnc. rewrite fields_of_M'.
           destruct (changed_of_node changed (Node_variable y));
             [ cbn in Hnc; discriminate Hnc | exact Hy ].
        -- exact (keyed_lookup_in field_eqb field_eqb_decides _ _ fy Hny).
    + (* The reads of the new slots of an unboxed value. *)
      unfold unboxed_slot_read_ok, slot_read_field in Hslot.
      apply andb_true_iff in Hslot as [Hk Hslot].
      apply kind_eqb_eq in Hk. subst kind.
      destruct (field_read d) as [[f arg] | ] eqn:Hfr; [ | discriminate Hslot ].
      destruct f as [ | w | | | | | | | ]; try discriminate Hslot.
      destruct arg as [y | | ]; try discriminate Hslot.
      destruct (fields_of unboxed y) eqn:Hyf; [ discriminate Hslot | ].
      unfold changed_name in Hslot.
      destruct (changed_of_node changed (Node_variable y)) as [r | ] eqn:Hyc;
        [ | discriminate Hslot ].
      destruct (Keyed.lookup field_eqb (Field_value_slot w) (cr_fields r))
        as [[sl | nested] | ] eqn:Hl; try discriminate Hslot.
      cbn [option_map] in Hslot.
      apply andb_true_iff in Hslot as [Hne Halike].
      unfold fields_alike in Halike.
      destruct (unboxed_field_pairs Value (Unboxed fields) (Unboxed nested))
        as [pairs | ] eqn:Hp; [ | discriminate Halike ].
      assert (Hyfr : fresh y)
        by (apply Hdf; destruct (field_read_slot d w (Var y) Hfr)
              as [-> | (fs & ->)]; left; reflexivity).
      destruct (field_read_slot d w (Var y) Hfr) as [-> | (fs & ->)];
        cbn beta iota;
        (destruct (is_dead_var sol x Value); [ apply rr_invalid | ]);
        (destruct (is_var_used sol x Value); [ | exact IHrr ]);
        cbn iota; unfold rebuild_unboxed_binding, load_unboxed_field;
        cbn [unboxed_of changed_of];
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y);
        rewrite Hyf, Hyc, Hl, Hp;
        (eapply rr_proj_unboxed; [ exact HxM | exact HxCH | exact HxF
                                 | reflexivity | | | | exact Hp
                                 | exact IHrr ]);
        [ cbn; destruct (is_var_used sol y _); [ left; reflexivity | ];
          right; eexists; reflexivity
        | rewrite (CHf_var unboxed changed y Hyfr); exact Hyc
        | exact (keyed_lookup_in field_eqb field_eqb_decides _ _ _ Hl)
        | cbn; destruct (is_var_used sol y _); [ left; reflexivity | ];
          right; eexists; reflexivity
        | rewrite (CHf_var unboxed changed y Hyfr); exact Hyc
        | exact (keyed_lookup_in field_eqb field_eqb_decides _ _ _ Hl) ].
  - (* A changed variable: a set whose representation changes. *)
    destruct d as [ | | | slot decls vs | | | ]; try discriminate Hbind.
    unfold changed_set_ok in Hbind.
    apply andb_true_iff in Hbind as [Hbind Hfields].
    apply andb_true_iff in Hbind as [Hbind Hdist].
    apply andb_true_iff in Hbind as [Hbind Hproj].
    apply andb_true_iff in Hbind as [Hbind Hlk].
    apply andb_true_iff in Hbind as [Hk Hslot].
    apply kind_eqb_eq in Hk. subst kind.
    apply Pos.eqb_eq in Hslot. subst slot.
    apply option_code_eqb_eq in Hlk.
    destruct (projections_of_set x (Closure_of_set (cr_current r) decls vs)
                body)
      eqn:Hp; [ | discriminate Hproj ].
    cbn [existsb orb]. rewrite Hun. cbn [is_none negb orb].
    destruct (is_dead_var sol x Value); [ apply rr_invalid | ].
    destruct (is_var_used sol x Value); [ | exact IHrr ].
    unfold rebuild_set_u. rewrite Hxc. cbn [rebuild_set].
    change (rename_function_slot r (cr_current r)) with (current_slot r).
    assert (Hvz : ∀ z, z ∈ variables_in_slots vs -> fresh z)
      by (intros z Hz; apply Hdf; exact Hz).
    destruct (changed_slots en (ue_env enu) x r vs Hfields Hvz)
      as [Heq Hok].
    rewrite Heq.
    apply rr_set.
    + rewrite (CHf_var unboxed changed x Hxf). exact Hxc.
    + exact Hlk.
    + exact Hok.
    + unfold new_slots. rewrite (new_slots_keys _ _ Hok).
      apply distinct_NoDup. exact Hdist.
    + exact IHrr.
  - (* A plain variable. *)
    assert (Hplainx : plain_variable MM x = true).
    { rewrite plain_variable_M', Hxc, andb_true_r. unfold plain_variable.
      rewrite Hxf. unfold Keyed.is_bound. unfold fields_of in Hx.
      rewrite Hx. reflexivity. }
    apply orb_true_iff in Hbind as [Hbind | Hslot];
      [ apply orb_true_iff in Hbind as [Hb | Hset] | ].
    + (* A simple, a primitive, or a read of an unboxed name. *)
      apply andb_true_iff in Hb as [Hpb Hnc].
      pose proof (strong_plain_binding unboxed x kind d Hpb Hdf) as Hpb'.
      destruct (plain_binding_cases sol unboxed x kind d Hpb')
        as [Hnp | (p & y & f & fs & -> & Hread & Hreads & Hnamed & Hy
                   & Hkindc)].
      * destruct d as [s | p args | | | | | ];
          try (cbn in Hnp; discriminate Hnp);
          cbn beta iota;
          (destruct (is_dead_var sol x kind); [ apply rr_invalid | ]);
          (destruct (is_var_used sol x kind); [ | exact IHrr ]);
          cbn iota;
          rewrite (rebuild_let_u_plain_c (ue_env enu) x kind _ _ Hnp Hnc);
          (apply rr_let; [ exact Hplainx | | | exact IHrr ]);
          try (apply named_plain_rebuilt; exact (named_plain_MM _ Hnp Hnc));
          apply rebuild_named_not_heap;
          eapply FlambdaReaperFlowProofs.named_edges_not_heap;
          exact Hnamed_edges.
      * cbn beta iota.
        destruct (is_dead_var sol x kind) eqn:Hdead; [ apply rr_invalid | ].
        destruct (is_var_used sol x kind) eqn:Hused; [ | exact IHrr ].
        cbn iota. rewrite (Hnamed (ue_env enu)).
        assert (Hacc : g_accessor (graph_of_facts F) (Node_variable x) f
                         (Node_variable y)).
        { clear -Hread Hnamed_edges.
          destruct p; try discriminate Hread; simpl in Hread;
            injection Hread as <-; exact (proj1 (proj2 Hnamed_edges)). }
        destruct (reads_found x y kind f fs Hx Hy Hacc
                    (read_block_like p y f Hread) Hdead Hused Hkindc)
          as (v & Hv).
        unfold rebuild_let_u.
        assert (Hfr : field_read (Prim (Unary p) [Var y]) = Some (f, Var y))
          by (clear -Hread; destruct p; try discriminate Hread; exact Hread).
        rewrite Hfr. cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hv.
        apply (rr_read MM CH CSf RCHf D x kind
                 (Prim (Unary p) [rewrite_simple sol (ue_env enu) (Var y)]) f
                 (rewrite_simple sol (ue_env enu) (Var y)) y v fs).
        -- exact Hplainx.
        -- apply Hreads.
        -- simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
           right. eexists. reflexivity.
        -- rewrite fields_of_M'. unfold no_changed in Hnc.
           cbn [named_simples forallb not_changed] in Hnc.
           unfold changed_name in Hnc.
           destruct (changed_of_node changed (Node_variable y));
             [ cbn in Hnc; discriminate Hnc | exact Hy ].
        -- exact (keyed_lookup_in field_eqb field_eqb_decides f _ fs Hv).
        -- exact IHrr.
    + (* A set of closures that does not change. *)
      destruct d as [ | | | slot decls vs | | | ]; try discriminate Hset.
      unfold plain_set_ok in Hset. apply andb_true_iff in Hset as [Hvs Hproj].
      destruct (projections_of_set x (Closure_of_set slot decls vs) body)
        eqn:Hp; [ | discriminate Hproj ].
      cbn [existsb orb]. rewrite Hun. cbn [is_none negb orb].
      destruct (is_dead_var sol x kind); [ apply rr_invalid | ].
      destruct (is_var_used sol x kind); [ | exact IHrr ].
      unfold rebuild_set_u. rewrite Hxc. cbn [rebuild_set].
      destruct (rebuild_value_slots_c en (ue_env enu)
                  (set_closure_nodes F (Node_variable x)) vs
                  (eq_sym (ci_kinds _ _ _ Hci)) Hvs Hdf)
        as [Heq Hplain].
      rewrite Heq. apply rr_plain_set; [ exact Hplainx | exact Hplain
                                        | exact IHrr ].
    + (* A read of a new slot. *)
      unfold plain_slot_read_ok, slot_read_field in Hslot.
      destruct (field_read d) as [[f arg] | ] eqn:Hfr; [ | discriminate Hslot ].
      destruct f as [ | w | | | | | | | ]; try discriminate Hslot.
      destruct arg as [y | | ]; try discriminate Hslot.
      destruct (fields_of unboxed y) eqn:Hyf; [ discriminate Hslot | ].
      unfold changed_name in Hslot.
      destruct (changed_of_node changed (Node_variable y)) as [r | ] eqn:Hyc;
        [ | discriminate Hslot ].
      destruct (Keyed.lookup field_eqb (Field_value_slot w) (cr_fields r))
        as [[sl | nested] | ] eqn:Hl; try discriminate Hslot.
      assert (Hyfr : fresh y)
        by (apply Hdf; destruct (field_read_slot d w (Var y) Hfr)
              as [-> | (fs & ->)]; left; reflexivity).
      destruct (field_read_slot d w (Var y) Hfr) as [-> | (fs & ->)];
        cbn beta iota;
        (destruct (is_dead_var sol x kind); [ apply rr_invalid | ]);
        (destruct (is_var_used sol x kind); [ | exact IHrr ]);
        cbn iota; unfold rebuild_let_u; rewrite Hfr;
        cbn [unboxed_of changed_of];
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y);
        rewrite Hyf, Hyc, Hl;
        (eapply rr_proj; [ exact Hplainx | reflexivity | | | | exact IHrr ]);
        [ cbn; destruct (is_var_used sol y _); [ left; reflexivity | ];
          right; eexists; reflexivity
        | rewrite (CHf_var unboxed changed y Hyfr); exact Hyc
        | exact (keyed_lookup_in field_eqb field_eqb_decides _ _ _ Hl)
        | cbn; destruct (is_var_used sol y _); [ left; reflexivity | ];
          right; eexists; reflexivity
        | rewrite (CHf_var unboxed changed y Hyfr); exact Hyc
        | exact (keyed_lookup_in field_eqb field_eqb_decides _ _ _ Hl) ].
Qed.

Lemma rebuild_rr :
  (∀ e : expr, rebuild_ok e)
  ∧ (∀ h : continuation_handler,
       ∀ (en : rebuild_env) (enu : unbox_env)
         (C : finite_map (list decision * bool)) (ctx : option code_id)
         (ce : finite_map (list node)) (is_exn : bool),
         cinv C en enu ->
         repr_handler_r F sol cannot_change code_deps unit_code_ids codes
           unboxed changed my_closures nr reach (env_kinds en) C
           (handler_decisions true sol unboxed is_exn h) h
         = true ->
         names_ok_h h ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True False ctx ce h ->
         conts_ok (continuations_in_handler h) ->
         ∀ D : list continuation,
           dead_in D (live_conts_handler F sol code_deps codes h) ->
         ∀ dead_check : bool,
           ∃ (ps : list variable) (md : handler_metadata) (hp hu : expr),
             rebuild_handler_cc F sol cannot_change code_deps pdc
               unit_code_ids codes ff types en dead_check (handler_mask sol
                 is_exn h)
               h
             = Handler ps md hp
             ∧ rebuild_handler_u true F sol cannot_change code_deps pdc
                 unit_code_ids codes unboxed ffu changed types enu dead_check
                 (handler_decisions true sol unboxed is_exn h) h
               = Handler ps md hu
             ∧ forallb (plain_variable MM) ps = true
             ∧ rr MM CH CSf RCHf D hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en enu C ctx ce Hci Hrepr Hnames Hedges Hconts D
      HD.
    apply (let_rebuilt C en enu ctx ce x kind d body Hci Hrepr Hnames Hedges
             D).
    cbn [repr_expr_r] in Hrepr. apply andb_true_iff in Hrepr as [_ Hbody].
    apply (IH (bind_variables en [(x, kind)]) (ubind_variables enu [(x, kind)])
             C ctx ce (cinv_bind C en enu _ Hci) Hbody).
    + intros z Hz. apply Hnames. right. apply in_or_app. right. exact Hz.
    + exact (proj2 Hedges).
    + exact Hconts.
    + exact HD.
  - (* Let_static *)
    intros bindings body IHcode IH en enu C ctx ce Hci Hrepr Hnames Hedges
      Hconts D HD.
    cbn [repr_expr_r] in Hrepr. apply andb_true_iff in Hrepr as [Hbs Hbody].
    destruct Hedges as [Hb_edges Hbody_edges].
    unfold names_ok in Hnames. cbn [variables_of] in Hnames.
    cbn [continuations_of] in Hconts. cbn [live_conts] in HD.
    rewrite cc_let_static, u_let_static. constructor.
    + apply (bindings_rebuilt C en enu bindings); try assumption.
      * intros z Hz. apply Hnames, in_or_app. left. exact Hz.
      * eapply conts_ok_incl; [ | exact Hconts ]. intros k Hk.
        apply in_or_app. left. exact Hk.
    + apply (IH (bind_variables en []) (ubind_variables enu []) C ctx ce
               (cinv_bind C en enu [] Hci)).
      * exact Hbody.
      * intros z Hz. apply Hnames, in_or_app. right. exact Hz.
      * exact Hbody_edges.
      * eapply conts_ok_incl; [ | exact Hconts ]. intros k Hk.
        apply in_or_app. right. exact Hk.
      * eapply dead_in_incl; [ | exact HD ]. intros k Hk.
        apply in_or_app. right. exact Hk.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en enu C ctx ce Hci Hrepr Hnames Hedges Hconts D
      HD.
    cbn [repr_expr_r] in Hrepr. cbv zeta in Hrepr.
    cbn [continuations_of] in Hconts. cbn [live_conts] in HD.
    apply andb_true_iff in Hrepr as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    unfold names_ok in Hnames. cbn [variables_of] in Hnames.
    set (is_exn := handler_is_exn_handler h) in *.
    set (ds := handler_decisions true sol unboxed is_exn h) in *.
    assert (Hfacts : map is_kept_decision ds = handler_mask sol is_exn h
                     ∧ forallb decided (normed ds) = true).
    { unfold ds. destruct h as [params md hb]. cbn [repr_handler_r] in Hh.
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [Hh Hdec].
      apply andb_true_iff in Hh as [Hlen _]. apply Nat.eqb_eq in Hlen.
      destruct (handler_params_c is_exn params md hb Hlen Hdec)
        as (_ & _ & Hkept).
      split; assumption. }
    destruct Hfacts as [Hkept Hdec].
    pose proof (cinv_conts C en enu [(k, ds)] Hci
                  (fun k0 ds0 (H : (k0, ds0) ∈ [(k, ds)]) =>
                     match H with
                     | or_introl Heq =>
                       eq_ind ds (fun d => forallb decided (normed d) = true)
                         Hdec ds0
                         (f_equal snd Heq)
                     | or_intror H' => match H' with end
                     end))
      as Hci'.
    cbn [map app] in Hci'. rewrite Hkept in Hci'.
    assert (Hcb : conts_ok (continuations_of body))
      by (eapply conts_ok_incl; [ | exact Hconts ]; intros k0 H; right;
          apply in_or_app; right; exact H).
    assert (Hch : conts_ok (continuations_in_handler h))
      by (eapply conts_ok_incl; [ | exact Hconts ]; intros k0 H; right;
          apply in_or_app; left; exact H).
    assert (HDh : dead_in D (live_conts_handler F sol code_deps codes h))
      by (eapply dead_in_incl; [ | exact HD ]; intros k0 H;
          apply in_or_app; left; exact H).
    pose proof (IH _ _ _ ctx ((k, FlambdaReaperTraversal.handler_params h)
                                :: ce) Hci' Hbody
                  (fun z H => Hnames z (in_or_app _ _ _ (or_intror H)))
                  Hbody_edges Hcb (without [k] D))
      as IHrr.
    assert (HDb : dead_in (without [k] D)
                    (live_conts F sol code_deps codes body)).
    { intros k0 Hk0 Hl. apply in_without_iff in Hk0 as [Hk0 Hne].
      apply (HD k0 Hk0). apply in_or_app. right. apply in_without_iff.
      split; assumption. }
    specialize (IHrr HDb).
    rewrite cc_let_cont, u_let_cont. cbv zeta. fold is_exn ds.
    destruct (proj1 (rr_shape MM CH CSf RCHf) _ _ _ IHrr) as [Hfc _].
    match goal with
    | |- rr _ _ _ _ _ (if among k (free_continuations_of_expr ?p) then _ else _)
           (if among k (free_continuations_of_expr ?q) then _ else _) =>
      destruct (among k (free_continuations_of_expr q)) eqn:Hku;
      [ assert (Hkc : among k (free_continuations_of_expr p) = true)
      | assert (Hnk : k ∉ free_continuations_of_expr q)
          by (intros Hin; apply among_In in Hin; rewrite Hin in Hku;
              discriminate Hku);
        destruct (among k (free_continuations_of_expr p)) ]
    end.
    + apply among_In. apply among_In in Hku.
      destruct (Hfc k Hku) as [Hin | Hin]; [ exact Hin | ].
      apply in_without_iff in Hin as [_ Hin]. exfalso. apply Hin.
      left. reflexivity.
    + rewrite Hkc.
      destruct (IHh en enu C ctx ce is_exn Hci Hh
                  (fun z H => Hnames z (in_or_app _ _ _ (or_introl H)))
                  Hh_edges Hch D HDh true)
        as (ps & md & hp & hu & Hp & Hu & Hps & Hhrr).
      fold ds in Hu. rewrite Hp, Hu. constructor; assumption.
    + apply rr_let_cont_never; [ exact Hnk | ].
      exact (proj1 (rr_dead_shadow MM CH CSf RCHf D _ _ k IHrr Hnk)).
    + exact (proj2 (rr_dead_shadow MM CH CSf RCHf D _ _ k IHrr Hnk)).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en enu C ctx ce Hci Hrepr Hnames Hedges
      Hconts Dd HDd.
    cbn [repr_expr_r] in Hrepr. cbv zeta in Hrepr.
    cbn [continuations_of] in Hconts. cbn [live_conts] in HDd.
    set (D' := without (map fst handlers) Dd).
    assert (Hcb : conts_ok (continuations_of body))
      by (eapply conts_ok_incl; [ | exact Hconts ]; intros k0 H;
          apply in_or_app; right; exact H).
    assert (HD' : dead_in D'
                    (flat_map (fun '(_, h) =>
                                 live_conts_handler F sol code_deps codes h)
                       handlers
                     ++ live_conts F sol code_deps codes body)).
    { intros k0 Hk0 Hl. apply in_without_iff in Hk0 as [Hk0 Hne].
      apply (HDd k0 Hk0). apply in_without_iff. split; assumption. }
    assert (HDb : dead_in D' (live_conts F sol code_deps codes body))
      by (eapply dead_in_incl; [ | exact HD' ]; intros k0 H;
          apply in_or_app; right; exact H).
    apply andb_true_iff in Hrepr as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    unfold names_ok in Hnames. cbn [variables_of] in Hnames.
    set (D := map (fun '(k, h) =>
                     (k, handler_decisions true sol unboxed false h))
                handlers).
    assert (HC : map (fun '(k, h) =>
                        (k,
                         (normed (handler_decisions true sol unboxed false h),
                          true)))
                   handlers
                 = map (fun '(k, ds) => (k, (normed ds, true))) D)
      by (unfold D; rewrite map_map; apply map_ext; intros [k h];
          reflexivity).
    rewrite HC in Hhs, Hbody.
    assert (Hfacts : ∀ k h, (k, h) ∈ handlers ->
              map is_kept_decision (handler_decisions true sol unboxed false h)
              = handler_mask sol false h
              ∧ forallb decided
                  (normed (handler_decisions true sol unboxed false h))
                = true).
    { intros k [params md hb] Hin. specialize (Hhs _ Hin).
      cbn beta iota in Hhs. cbn [repr_handler_r] in Hhs.
      apply andb_true_iff in Hhs as [Hh _].
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [Hh Hdec].
      apply andb_true_iff in Hh as [Hlen _]. apply Nat.eqb_eq in Hlen.
      destruct (handler_params_c false params md hb Hlen Hdec)
        as (_ & _ & Hkept).
      split; assumption. }
    assert (Hmasks : map (fun '(k, h) => (k, handler_mask sol false h))
                       handlers
                     = map (fun '(k, ds) => (k, map is_kept_decision ds)) D).
    { unfold D. rewrite map_map. apply map_ext_in. intros [k h] Hin.
      rewrite (proj1 (Hfacts k h Hin)). reflexivity. }
    assert (HD : ∀ k ds, (k, ds) ∈ D -> forallb decided (normed ds) = true).
    { intros k ds Hin. unfold D in Hin.
      apply in_map_iff in Hin as ([k' h] & Heq & Hin).
      injection Heq as <- <-. exact (proj2 (Hfacts k' h Hin)). }
    pose proof (cinv_conts C en enu D Hci HD) as Hci'.
    rewrite <- Hmasks in Hci'.
    set (C' := map (fun '(k, ds) => (k, (normed ds, true))) D ++ C) in *.
    set (en' := bind_continuations en
                  (map (fun '(k, h) => (k, handler_mask sol false h))
                     handlers)) in *.
    set (enu' := ubind_continuations enu D) in *.
    pose proof (IH en' enu' C' ctx
                  (FlambdaReaperFlowProofs.rec_conts ce handlers) Hci' Hbody
                  (fun z H => Hnames z (in_or_app _ _ _ (or_intror H)))
                  Hbody_edges Hcb D' HDb)
      as IHrr.
    assert (Hhandlers :
      ∀ hs : list (continuation * continuation_handler),
        (∀ k h, (k, h) ∈ hs -> (k, h) ∈ handlers) ->
        rr_handlers MM CH CSf RCHf D'
          (map (fun '(k, h) =>
                  (k, rebuild_handler_cc F sol cannot_change code_deps pdc
                        unit_code_ids codes ff types en' false
                        (handler_mask sol false h) h))
             hs)
          (map (fun '(k, h) =>
                  (k, rebuild_handler_u true F sol cannot_change
                        code_deps pdc unit_code_ids codes unboxed ffu changed
                          types
                        enu' false (handler_decisions true sol unboxed false h)
                        h))
             hs)).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; [ constructor | ].
      assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
      destruct (IHh k h Hin en' enu' C' ctx
                  (FlambdaReaperFlowProofs.rec_conts ce handlers) false Hci'
                  (Hhs _ Hin))
        with (dead_check := false) (D := D')
        as (ps & md & hp & hu & Hp & Hu & Hps & Hhrr).
      { intros z Hz. apply Hnames, in_or_app. left. apply in_flat_map.
        exists (k, h). split; [ exact Hin | exact Hz ]. }
      { exact (every_in _ _ (k, h) Hhs_edges Hin). }
      { eapply conts_ok_incl; [ | exact Hconts ]. intros k0 Hk0.
        apply in_or_app. left. apply in_flat_map.
        exists (k, h). split; [ exact Hin | right; exact Hk0 ]. }
      { eapply dead_in_incl; [ | exact HD' ]. intros k0 Hk0.
        apply in_or_app. left. apply in_flat_map.
        exists (k, h). split; [ exact Hin | exact Hk0 ]. }
      cbn [map]. rewrite Hp, Hu. constructor; [ exact Hps | exact Hhrr | ].
      apply IHhs. intros k' h' H'. apply Hsub. right. exact H'. }
    rewrite cc_let_cont_rec, u_let_cont_rec. cbv zeta. fold en'.
    fold D. fold enu'.
    apply rr_let_cont_rec;
      (rewrite map_map;
       replace (map (fun x => fst (let '(k, h) := x in _)) handlers)
         with (map fst handlers)
         by (apply map_ext; intros [k h]; reflexivity));
      [ apply Hhandlers; auto | exact IHrr ].
  - (* Apply *)
    intros ap en enu C ctx ce Hci Hrepr Hnames _ Hconts D HD.
    cbn [continuations_of continuations_in_apply] in Hconts.
    exact (call_rebuilt_r C en enu ap D Hci Hrepr Hnames
             (proj1 (Hconts _ (or_intror (or_introl eq_refl))))
             (proj2 (Hconts _ (or_intror (or_introl eq_refl)))) HD).
  - (* Apply_cont *)
    intros ac en enu C ctx ce Hci Hrepr Hnames _ _ D HD.
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    exact (apply_cont_rebuilt C en enu ac D Hci Hrepr Hnames HD).
  - (* Switch *)
    intros scrutinee arms en enu C ctx ce Hci Hrepr Hnames _ _ D HD.
    cbn [repr_expr_r] in Hrepr. apply andb_true_iff in Hrepr as [Hs Harms].
    unfold cc_rebuild, u_rebuild. cbn [rebuild_expr_cc rebuild_expr_u].
    exact (switch_rebuilt C en enu scrutinee arms Hci Hs Harms Hnames D HD).
  - (* Invalid *)
    intros. constructor.
  - (* Handler *)
    intros params md handler IH en enu C ctx ce is_exn Hci Hrepr Hnames
      Hedges Hconts D HD dc.
    cbn [repr_handler_r] in Hrepr.
    apply andb_true_iff in Hrepr as [Hrepr Hbody].
    apply andb_true_iff in Hrepr as [Hrepr Hps].
    apply andb_true_iff in Hrepr as [Hrepr Hdec].
    apply andb_true_iff in Hrepr as [Hlen _].
    apply Nat.eqb_eq in Hlen.
    destruct (handler_params_c is_exn params md handler Hlen Hdec)
      as (Hfst & Hsnd & Hkept).
    cbn [rebuild_handler_cc]. rewrite u_rebuild_handler_c. cbv zeta.
    rewrite (u_handler_metadata_decided is_exn params md handler Hlen Hdec).
    rewrite Hfst.
    eexists _, _, _, _. split; [ reflexivity | split; [ reflexivity | ] ].
    unfold names_ok_h in Hnames. cbn [variables_in_handler] in Hnames.
    split.
    + apply filter_kept_forallb. apply forallb_forall. intros x Hx.
      apply (plain_c_var_M' unboxed changed x).
      * exact (proj1 (forallb_forall _ params) Hps x Hx).
      * apply Hnames, in_or_app. left. exact Hx.
    + destruct (dc && _); [ constructor | ].
      apply (IH (bind_variables en (with_kinds params (hm_params_kinds md)))
               (ubind_variables enu (with_kinds params (hm_params_kinds md)))
               C ctx ce).
      * apply cinv_bind. exact Hci.
      * cbn [env_kinds bind_variables bind_variables_then].
        rewrite (with_kinds_combine params _ Hlen). exact Hbody.
      * intros z Hz. apply Hnames, in_or_app. right. exact Hz.
      * exact Hedges.
      * exact Hconts.
      * exact HD.
Qed.
End Rebuilds.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The [my_closure] of the code that a unit binds is a variable of the
   unit. *)
Lemma bound_my_closures_in :
  (∀ e : expr, ∀ (id : code_id) (mc : variable),
     (id, mc) ∈ bound_code_my_closures e -> mc ∈ variables_of e)
  ∧ (∀ h : continuation_handler, ∀ (id : code_id) (mc : variable),
       (id, mc) ∈ handler_bound_code_my_closures h ->
       mc ∈ variables_in_handler h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [bound_code_my_closures handler_bound_code_my_closures variables_of
         variables_in_handler];
    try (intros; contradiction).
  - intros x kind d body IH id mc H. right. apply in_or_app. right.
    exact (IH id mc H).
  - intros bindings body _ IH id mc H. apply in_app_or in H as [H | H];
      apply in_or_app; [ left | right; exact (IH id mc H) ].
    apply in_flat_map in H as (b & Hb & H). apply in_flat_map.
    exists b. split; [ exact Hb | ].
    destruct b as [id' c | | ]; [ | destruct H | destruct H ].
    destruct H as [Heq | []]. injection Heq as <- <-.
    destruct c. cbn. left. reflexivity.
  - intros k h body IHh IH id mc H. apply in_app_or in H as [H | H];
      apply in_or_app; [ left; exact (IHh id mc H)
                       | right; exact (IH id mc H) ].
  - intros handlers body IHh IH id mc H. apply in_app_or in H as [H | H];
      apply in_or_app; [ left | right; exact (IH id mc H) ].
    apply in_flat_map in H as ([k h] & Hin & H). apply in_flat_map.
    exists (k, h). split; [ exact Hin | exact (IHh k h Hin id mc H) ].
  - intros params md handler IH id mc H. apply in_or_app. right.
    exact (IH id mc H).
Qed.

(* ======================================================================== *)
(* The fragment of FlambdaReaperUnboxingRefinesClosuresSpec.v is within the *)
(* wider one                                                                *)
(* ======================================================================== *)

Section Within.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.

Abbreviation old_expr kinds C e :=
  (repr_expr F sol cannot_change code_deps unit_code_ids codes unboxed changed
     my_closures nr kinds C e).
Abbreviation new_expr kinds C e :=
  (repr_expr_w F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr kinds C e).

(* [C'] describes the continuations as [C] does, but the exception
   continuations of code, which it may describe as raised to. *)
Definition conts_within (C C' : finite_map (list decision * bool)) : Prop :=
  ∀ k : continuation,
    (lookup k C = None -> lookup k C' = None)
    ∧ (∀ ds, lookup k C = Some (ds, true) -> lookup k C' = Some (ds, true))
    ∧ (∀ ds, lookup k C = Some (ds, false) ->
         ∃ ds', lookup k C' = Some (ds', false)).

Lemma conts_within_app (L C C' : finite_map (list decision * bool)) :
  conts_within C C' -> conts_within (L ++ C) (L ++ C').
Proof.
  intros H k. rewrite !lookup_app.
  destruct (lookup k L) as [[ds b] | ]; [ | exact (H k) ].
  split; [ intros Hn; discriminate Hn | ].
  split; [ intros ds0 Hs; exact Hs | intros ds0 Hs; exists ds0; exact Hs ].
Qed.

Lemma returns_within (C C' : finite_map (list decision * bool))
    (k : continuation) (rd : list decision) :
  conts_within C C' -> returns_ok C k rd = true -> returns_ok C' k rd = true.
Proof.
  intros HC H. unfold returns_ok in *. destruct (HC k) as (Hn & Ht & _).
  destruct (lookup k C) as [[ds [ | ]] | ] eqn:E.
  - rewrite (Ht ds eq_refl). exact H.
  - discriminate H.
  - rewrite (Hn eq_refl). exact H.
Qed.

Lemma apply_cont_within (C C' : finite_map (list decision * bool))
    (ac : apply_cont) :
  conts_within C C' -> apply_cont_ok_c unboxed changed C ac = true ->
  apply_cont_ok_w unboxed changed C' ac = true.
Proof.
  intros HC H. unfold apply_cont_ok_c, apply_cont_ok_w in *.
  apply andb_true_iff in H as [Ha Hk]. rewrite Ha. cbn [andb].
  destruct (HC (ac_continuation ac)) as (Hn & Ht & _).
  destruct (lookup (ac_continuation ac) C) as [[ds [ | ]] | ] eqn:E.
  - rewrite (Ht ds eq_refl). destruct ds; reflexivity.
  - discriminate Hk.
  - rewrite (Hn eq_refl). reflexivity.
Qed.

Lemma call_within (C C' : finite_map (list decision * bool))
    (ap : apply_expr) :
  conts_within C C' ->
  repr_call F sol cannot_change code_deps unit_code_ids codes unboxed changed
    my_closures nr C ap
  = true ->
  repr_call_w F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr C' ap
  = true.
Proof.
  intros HC H. unfold repr_call, repr_call_w in *.
  destruct (ap_callee ap) as [f | | ]; try discriminate H.
  destruct (code_id_actually_called F sol ap) as [id | ];
    [ | discriminate H ].
  apply andb_true_iff in H as [Hkc H]. rewrite Hkc. cbn [andb].
  destruct (lookup id codes) as [md | ]; [ | discriminate H ].
  destruct (lookup id my_closures) as [mc | ]; [ | discriminate H ].
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md);
    [ discriminate H | ].
  cbv beta iota.
  apply andb_true_iff in H as [H Hret].
  apply andb_true_iff in H as [H Hextra].
  apply andb_true_iff in H as [H Hargs].
  apply andb_true_iff in H as [H Hrlen].
  apply andb_true_iff in H as [H Hra].
  apply andb_true_iff in H as [H Haa].
  apply andb_true_iff in H as [H Hfu].
  apply andb_true_iff in H as [H Hcl].
  apply andb_true_iff in H as [H Hmcd].
  apply andb_true_iff in H as [H Hnr].
  apply andb_true_iff in H as [Hpd Hrd].
  assert (Hx : extra_args_ok unboxed changed C' ap = true).
  { unfold extra_args_ok, exn_extra_simples.
    destruct (ap_exn_extra_args ap); [ reflexivity | discriminate Hextra ]. }
  rewrite Hnr, Hcl, Hfu, Haa, Hra, Hargs, Hx, Hpd, Hrd, Hmcd, Hrlen.
  cbn [andb].
  destruct (code_never_returns sol code_deps id); [ reflexivity | ].
  exact (returns_within C C' _ _ HC Hret).
Qed.

Lemma code_within (id : code_id) (c : code) :
  (∀ kinds C C', conts_within C C' -> old_expr kinds C (code_body c) = true ->
     new_expr kinds C' (code_body c) = true) ->
  repr_code F sol cannot_change code_deps unit_code_ids codes unboxed changed
    my_closures nr id c
  = true ->
  repr_code_w F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr id c
  = true.
Proof.
  intros IH H. destruct c as [params mc ret exn md body].
  cbn [repr_code repr_code_w code_body] in *.
  apply andb_true_iff in H as [H Hlive]. rewrite Hlive, andb_true_r.
  apply andb_true_iff in H as [H Hch].
  apply andb_true_iff in H as [Hmd Hmc]. rewrite Hmd, Hmc. cbn [andb].
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md);
    [ discriminate Hch | ].
  cbv beta iota.
  apply andb_true_iff in Hch as [Hch Hbody].
  apply andb_true_iff in Hch as [Hch Hre].
  apply andb_true_iff in Hch as [Hch Hmcu].
  apply andb_true_iff in Hch as [Hch Hmcd].
  apply andb_true_iff in Hch as [Hch Hps].
  apply andb_true_iff in Hch as [Hch Hrlen].
  apply andb_true_iff in Hch as [Hch Hplen].
  apply andb_true_iff in Hch as [Hch Hlen].
  apply andb_true_iff in Hch as [Hpd Hrd].
  rewrite Hps, Hmcu, Hre, Hpd, Hrd, Hlen, Hplen, Hrlen, Hmcd. cbn [andb].
  refine (IH _ _ _ _ Hbody).
  intros k. rewrite !lookup_cons.
  destruct (Pos.eqb k ret).
  - split; [ intros Hn; discriminate Hn | ].
    split; [ intros ds Hs; exact Hs | intros ds Hs; discriminate Hs ].
  - destruct (Pos.eqb k exn).
    + split; [ intros Hn; discriminate Hn | ].
      split; [ intros ds Hs; discriminate Hs | ].
      intros ds _. eexists. reflexivity.
    + split; [ intros Hn; exact Hn | ].
      split; [ intros ds Hs; discriminate Hs | intros ds Hs ].
      discriminate Hs.
Qed.

Lemma repr_within :
  (∀ e : expr, ∀ kinds C C',
     conts_within C C' -> old_expr kinds C e = true ->
     new_expr kinds C' e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C C' ds,
       conts_within C C' ->
       repr_handler F sol cannot_change code_deps unit_code_ids codes
         unboxed changed my_closures nr kinds C ds h
       = true ->
       repr_handler_w F sol cannot_change code_deps unit_code_ids codes
         unboxed changed my_closures nr kinds C' ds h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH kinds C C' HC H.
    cbn [repr_expr repr_expr_w] in *.
    apply andb_true_iff in H as [Hh Hb].
    rewrite Hh, (IH _ _ _ HC Hb). reflexivity.
  - (* Let_static *)
    intros bindings body IHcode IH kinds C C' HC H.
    cbn [repr_expr repr_expr_w] in *.
    apply andb_true_iff in H as [Hbs Hb].
    rewrite (IH _ _ _ HC Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [id c | id | s const];
      cbn [repr_binding repr_binding_w] in *; [ | reflexivity | exact Hbs ].
    exact (code_within id c (IHcode id c Hin) Hbs).
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH kinds C C' HC H.
    cbn [repr_expr repr_expr_w] in *. cbv zeta in *.
    apply andb_true_iff in H as [Hh Hb].
    assert (Hdec : forallb decided
                     (handler_decisions true sol unboxed
                        (handler_is_exn_handler h) h)
                   = true).
    { destruct h as [params md hb]. cbn [repr_handler] in Hh.
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [_ Hdec]. exact Hdec. }
    set (ds := handler_decisions true sol unboxed (handler_is_exn_handler h)
                 h) in *.
    assert (HC2 : conts_within ((k, (ds, true)) :: C)
                    ((k, (normed ds, true)) :: C')).
    { rewrite (normed_decided _ Hdec). exact (conts_within_app [_] _ _ HC). }
    apply andb_true_iff. split; [ exact (IHh _ _ _ _ HC Hh) | ].
    exact (IH _ _ _ HC2 Hb).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH kinds C C' HC H.
    cbn [repr_expr repr_expr_w] in *. cbv zeta in *.
    apply andb_true_iff in H as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hmap :
      map (fun '(k, h) =>
             (k, (normed (handler_decisions true sol unboxed false h), true)))
        handlers
      = map (fun '(k, h) =>
               (k, (handler_decisions true sol unboxed false h, true)))
          handlers).
    { apply map_ext_in. intros [k [params md hb]] Hin.
      specialize (Hhs _ Hin). cbn beta iota in Hhs.
      cbn [repr_handler] in Hhs.
      apply andb_true_iff in Hhs as [Hh _].
      apply andb_true_iff in Hh as [Hh _].
      apply andb_true_iff in Hh as [_ Hdec].
      rewrite (normed_decided _ Hdec). reflexivity. }
    rewrite Hmap.
    assert (HC' := conts_within_app
                     (map (fun '(k, h) =>
                             (k, (handler_decisions true sol unboxed false h,
                                  true)))
                        handlers)
                     C C' HC).
    apply andb_true_iff. split; [ | exact (IH _ _ _ HC' Hb) ].
    apply forallb_forall. intros [k h] Hin.
    exact (IHh k h Hin _ _ _ _ HC' (Hhs _ Hin)).
  - (* Apply *)
    intros ap kinds C C' HC H. cbn [repr_expr repr_expr_w] in *.
    exact (call_within C C' ap HC H).
  - (* Apply_cont *)
    intros ac kinds C C' HC H. cbn [repr_expr repr_expr_w] in *.
    exact (apply_cont_within C C' ac HC H).
  - (* Switch *)
    intros scrutinee arms kinds C C' HC H. cbn [repr_expr repr_expr_w] in *.
    apply andb_true_iff in H as [Hs Harms]. rewrite Hs. cbn [andb].
    rewrite forallb_forall in Harms |- *. intros [i ac] Hin.
    exact (apply_cont_within C C' ac HC (Harms _ Hin)).
  - (* Invalid *)
    intros kinds C C' _ _. reflexivity.
  - (* Handler *)
    intros params md handler IH kinds C C' ds HC H.
    cbn [repr_handler repr_handler_w] in *.
    apply andb_true_iff in H as [H Hb].
    apply andb_true_iff in H as [H Hps].
    apply andb_true_iff in H as [H Hdec].
    rewrite H, (normed_decided _ Hdec), Hdec, Hps. cbn [andb].
    exact (IH _ _ _ HC Hb).
Qed.

(* The fragment of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v is
   within that of FlambdaReaperUnboxingRefinesClosuresReachSpec.v, for any
   flow: [repr_call_r] holds every call of [repr_call_w]. *)
Variable reach : bool -> variable -> code_id -> bool.

Abbreviation reach_expr kinds C e :=
  (repr_expr_r F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr reach kinds C e).

Lemma code_within_r (id : code_id) (c : code) :
  (∀ kinds C, new_expr kinds C (code_body c) = true ->
     reach_expr kinds C (code_body c) = true) ->
  repr_code_w F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr id c
  = true ->
  repr_code_r F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr reach id c
  = true.
Proof.
  intros IH H. destruct c as [params mc ret exn md body].
  cbn [repr_code_w repr_code_r code_body] in *.
  apply andb_true_iff in H as [H Hlive]. rewrite Hlive, andb_true_r.
  apply andb_true_iff in H as [Hpre Hch]. rewrite Hpre. cbn [andb].
  destruct (code_change_u sol cannot_change code_deps unit_code_ids unboxed
              id md).
  - destruct (lookup id code_deps) as [cd | ];
      [ | rewrite andb_false_r in Hch; discriminate Hch ].
    cbv zeta in *.
    apply andb_true_iff in Hch as [Htup Hch]. rewrite Htup. cbn [andb].
    apply andb_true_iff in Hch as [Hdec Hbody]. rewrite Hdec. cbn [andb].
    exact (IH _ _ Hbody).
  - apply andb_true_iff in Hch as [Hch Hbody]. rewrite Hch. cbn [andb].
    exact (IH _ _ Hbody).
Qed.

Lemma w_within_r :
  (∀ e : expr, ∀ kinds C,
     new_expr kinds C e = true -> reach_expr kinds C e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ds,
       repr_handler_w F sol cannot_change code_deps unit_code_ids codes
         unboxed changed my_closures nr kinds C ds h
       = true ->
       repr_handler_r F sol cannot_change code_deps unit_code_ids codes
         unboxed changed my_closures nr reach kinds C ds h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH kinds C H.
    cbn [repr_expr_w repr_expr_r] in *.
    apply andb_true_iff in H as [Hh Hb].
    rewrite Hh, (IH _ _ Hb). reflexivity.
  - (* Let_static *)
    intros bindings body IHcode IH kinds C H.
    cbn [repr_expr_w repr_expr_r] in *.
    apply andb_true_iff in H as [Hbs Hb].
    rewrite (IH _ _ Hb), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin.
    specialize (Hbs b Hin).
    destruct b as [id c | id | s const];
      cbn [repr_binding_w repr_binding_r] in *; [ | reflexivity | exact Hbs ].
    exact (code_within_r id c (IHcode id c Hin) Hbs).
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH kinds C H.
    cbn [repr_expr_w repr_expr_r] in *. cbv zeta in *.
    apply andb_true_iff in H as [Hh Hb].
    apply andb_true_iff. split; [ exact (IHh _ _ _ Hh) | exact (IH _ _ Hb) ].
  - (* Let_cont_recursive *)
    intros handlers body IHh IH kinds C H.
    cbn [repr_expr_w repr_expr_r] in *. cbv zeta in *.
    apply andb_true_iff in H as [Hhs Hb].
    rewrite (IH _ _ Hb), andb_true_r.
    rewrite forallb_forall in Hhs |- *. intros [k h] Hin.
    exact (IHh k h Hin _ _ _ (Hhs _ Hin)).
  - (* Apply *)
    intros ap kinds C H. cbn [repr_expr_w repr_expr_r] in *.
    unfold repr_call_r. rewrite H. reflexivity.
  - (* Apply_cont *)
    intros ac kinds C H. exact H.
  - (* Switch *)
    intros scrutinee arms kinds C H. exact H.
  - (* Invalid *)
    intros kinds C _. reflexivity.
  - (* Handler *)
    intros params md handler IH kinds C ds H.
    cbn [repr_handler_w repr_handler_r] in *.
    apply andb_true_iff in H as [H Hb]. rewrite H. cbn [andb].
    exact (IH _ _ Hb).
Qed.

End Within.

(* The unit of the fragment of FlambdaReaperUnboxingRefinesClosuresSpec.v is
   of the wider fragment. *)
Lemma repr_within_unit (nr : code_id -> bool) (is_local : field -> bool)
    (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) :
  repr_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr e
  = true ->
  repr_calls_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr e
  = true.
Proof.
  unfold repr_unboxing_nr, repr_calls_unboxing_nr. cbv zeta. intros H.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  refine (proj1 (repr_within _ _ _ _ _ _ _ _ _ _) e [] [] [] _ H).
  intros k. split; [ intros Hn; exact Hn | ].
  split; intros ds Hs; discriminate Hs.
Qed.

(* The unit of the fragment of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v is of that of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v, for any flow. *)
Lemma repr_calls_within_reach_unit (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (reach : bool -> variable -> code_id -> bool) (e : expr) :
  repr_calls_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr e
  = true ->
  repr_reach_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr reach e
  = true.
Proof.
  unfold repr_calls_unboxing_nr, repr_reach_unboxing_nr. cbv zeta. intros H.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  exact (proj1 (w_within_r _ _ _ _ _ _ _ _ _ _ reach) e [] [] H).
Qed.

(* The two rebuilt units of the fragment, for any rewriting of what the
   parameters kept state, are related by [rr], with the flow of the run of
   the first as [reach] says: the proof of the statement below up to its
   last step. *)
Lemma rebuild_repr_reach_modes_with_rr (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_reach_unboxing_nr nr is_local (traverse_unit flag u e) T
    cannot_change (unit_code_deps flag u e) dominated to_unbox tcr reach e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info),
    distinct (field_variables M) = true
    ∧ (∀ (x : variable) (r : closure_repr),
         CH x = Some r -> fields_of M x = Some [])
    ∧ (∀ (x : variable) (r : closure_repr),
         CH x = Some r -> among x (field_variables M) = false)
    ∧ rr M CH CS (fun b f id => reach b f id = true) []
        (rebuild_unit_cc_with types (traverse_unit flag u e)
           (solution_of_tables T) (fun n => mem_node n cannot_change)
           (unit_code_deps flag u e) pdc e)
        (rebuild_unit_unbox_repr_mode_with types true is_local
           (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
           dominated to_unbox tcr pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hrepr Hcheck.
  set (F := traverse_unit flag u e) in *.
  unfold repr_reach_unboxing_nr in Hrepr.
  apply andb_true_iff in Hrepr as [Hrepr Huses].
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  pose proof (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
                dominated to_unbox
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)))
    as Hnext.
  cbv zeta in Hnext. destruct Hnext as (Hnext & _ & _).
  unfold rebuild_unit_unbox_repr_mode_with, rebuild_unit_cc_with.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  assert (Hnext' : (FlambdaNameAbstractionSpec.fresh_name (names_of e)
                    <= next)%positive).
  { unfold unit_unboxed_fields in Hunboxed. rewrite Hunboxed in Hnext.
    exact Hnext. }
  clear Hnext.
  cbn [fst] in Hrepr, Huses, Hdistinct, Hfresh.
  set (changed := unit_changed_representations is_local F T dominated
                    to_unbox tcr e) in *.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    pose proof Hx as Hxu.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (FlambdaReaperUnboxingRefinesCallsProofs.add_usages_unboxed F T _
               unboxed _ _ true Huses Hxu)
      in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_block_like_field f = true ->
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hblock Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead.
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox
                    (FlambdaNameAbstractionSpec.fresh_name (names_of e)) a
                    (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & Hfields).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses' : mem_node (Node_variable y)
                       (add_usages_through_function_slots F T
                          (bound_code_my_closures e) true
                          (get_direct_usages T [Node_variable y]))
                     = true).
    { rewrite (FlambdaReaperUnboxingRefinesCallsProofs.add_usages_unboxed F T
                 _ unboxed _ _ true Huses Hy).
      apply mem_node_In. apply get_direct_usages_self; assumption. }
    subst fields.
    destruct (get_fields_found F T _ _ _ _ Hacc Huses' Hfs Hux)
      as [Hex Hall].
    apply (mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  assert (Hmc : ∀ (id : code_id) (mc : variable),
            lookup id (bound_code_my_closures e) = Some mc ->
            among mc (field_variables unboxed) = false).
  { intros id mc H. apply Hfresh.
    exact (proj1 bound_my_closures_in e id mc (lookup_in id mc _ H)). }
  pose proof (proj1 (rebuild_rr F (solution_of_tables T)
                       (fun n => mem_node n cannot_change)
                       (unit_code_deps flag u e) pdc
                       (static_names code_ids_bound e) (bound_code_metadata e)
                       unboxed changed (bound_code_my_closures e)
                       (FlambdaNameAbstractionSpec.fresh_name (names_of e))
                       next noted deps cds (unit_kept e) (unit_codes e)
                       (static_names symbols_of_binding e) is_local nr types
                       reach Hcl Hslots Hfields_read Hreads_found Hmc)
                e empty_env empty_unbox_env [] None (unit_conts u))
    as Hrr.
  exists (M' unboxed changed), (CHf unboxed changed). eexists.
  split; [ rewrite field_variables_M'; exact Hdistinct | ].
  split; [ exact (CHf_M unboxed changed) | ].
  split; [ exact (CHf_fresh unboxed changed) | ].
  apply Hrr.
  - split; try reflexivity.
    + intros k ds b H. discriminate H.
    + intros k _. split; reflexivity.
  - exact Hrepr.
  - intros z Hz. exact (Hfresh z Hz).
  - exact Hedges.
  - intros k Hk.
    assert (Hle : (k <= fold_right Pos.max 1 (names_of e))%positive)
      by (apply FlambdaReaperUnboxingRefinesProofs.fold_max_le;
          unfold names_of; apply in_or_app; right; exact Hk).
    unfold FlambdaNameAbstractionSpec.fresh_name in Hnext' |- *.
    split; intros ->; lia.
  - intros k [].
Qed.

(* The statement of the fix, for any rewriting of what the parameters
   kept state, the same for the two rebuilds: the handlers and the code of
   the fragment keep their parameters alike in the two, and state the
   same. A call whose code is not known calls code that the flow of the
   run of the first rebuilt unit brings to its callee, as [reach] says. *)
Lemma rebuild_repr_reach_modes_with_refines_cc (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_reach_unboxing_nr nr is_local (traverse_unit flag u e) T
    cannot_change (unit_code_deps flag u e) dominated to_unbox tcr reach e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  flow_run (fun b f id => reach b f id = true)
    (State
       (rebuild_unit_cc_with types (traverse_unit flag u e)
          (solution_of_tables T) (fun n => mem_node n cannot_change)
          (unit_code_deps flag u e) pdc e)
       [] [] empty_static_env empty_heap no_reads) ->
  program_refines
    (rebuild_unit_cc_with types (traverse_unit flag u e)
       (solution_of_tables T) (fun n => mem_node n cannot_change)
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode_with types true is_local
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox tcr pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hrepr Hcheck Hflow.
  destruct (rebuild_repr_reach_modes_with_rr types flag is_local u e T
              cannot_change dominated to_unbox tcr pdc nr reach Hslots
              Hfragment Hclosed Hrepr Hcheck)
    as (M & CH & CS & Hdistinct & HCM & HCf & Hrr).
  exact (rr_refines_flow M Hdistinct CH HCM HCf CS _ _ _ Hrr Hflow).
Qed.

(* The same for the fragment of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v, which is within the
   wider one whatever the flow: where the flow may bring any code
   anywhere. *)
Lemma rebuild_repr_calls_modes_with_refines_cc (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
    cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_with types (traverse_unit flag u e)
       (solution_of_tables T) (fun n => mem_node n cannot_change)
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode_with types true is_local
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox tcr pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hrepr Hcheck.
  apply (rebuild_repr_reach_modes_with_refines_cc types flag is_local u e T
           cannot_change dominated to_unbox tcr pdc nr (fun _ _ _ => true)
           Hslots Hfragment Hclosed
           (repr_calls_within_reach_unit nr is_local _ T cannot_change _
              dominated to_unbox tcr (fun _ _ _ => true) e Hrepr)
           Hcheck).
  apply flow_run_total. intros b f id. reflexivity.
Qed.

(* The same for the fragment of FlambdaReaperUnboxingRefinesClosuresSpec.v,
   which is within the wider one. *)
Lemma rebuild_repr_modes_with_refines_cc (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_unboxing_nr nr is_local (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox tcr e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_with types (traverse_unit flag u e)
       (solution_of_tables T) (fun n => mem_node n cannot_change)
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode_with types true is_local
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox tcr pdc e).
Proof.
  intros Hslots Hfragment Hclosed Hrepr Hcheck.
  exact (rebuild_repr_calls_modes_with_refines_cc types flag is_local u e T
           cannot_change dominated to_unbox tcr pdc nr Hslots Hfragment
           Hclosed (repr_within_unit nr is_local _ T cannot_change _ dominated
                      to_unbox tcr e Hrepr)
           Hcheck).
Qed.

(* The same without types. *)
Lemma rebuild_repr_modes_refines_cc
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_unboxing_nr nr is_local (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox tcr e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines
    (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) pdc e)
    (rebuild_unit_unbox_repr_mode true is_local
       (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
       dominated to_unbox tcr pdc e).
Proof.
  exact (rebuild_repr_modes_with_refines_cc None flag is_local u e T
           cannot_change dominated to_unbox tcr pdc nr).
Qed.

Theorem rebuild_repr_fixed_refines_cc_never_returns :
  rebuild_repr_fixed_refines_cc_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr.
  exact (rebuild_repr_modes_refines_cc flag is_local u e T
           cannot_change dominated to_unbox tcr pdc nr).
Qed.

Theorem rebuild_repr_fixed_refines_cc :
  rebuild_repr_fixed_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc.
  exact (rebuild_repr_fixed_refines_cc_never_returns flag is_local u e T
           cannot_change dominated to_unbox tcr pdc (fun _ => false)).
Qed.

Theorem rebuild_repr_fixed_tail_refines_cc :
  rebuild_repr_fixed_tail_refines_cc_statement.
Proof. exact rebuild_repr_fixed_refines_cc. Qed.

Theorem rebuild_repr_fixed_local_never_returns :
  rebuild_repr_fixed_local_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hrepr
    Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_fixed_refines_cc_never_returns flag is_local u e T
              cannot_change dominated to_unbox tcr pdc nr Hslots Hfrag Hclosed
              Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_fixed_observably_never_returns :
  rebuild_repr_fixed_observably_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hrepr
    Hcheck.
  exact (rebuild_repr_fixed_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox tcr pdc nr Hslots Hfrag Hany Hrk
           Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hrepr Hcheck).
Qed.

Theorem rebuild_repr_fixed_refines_observably_local :
  rebuild_repr_fixed_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc.
  exact (rebuild_repr_fixed_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox tcr pdc (fun _ => false)).
Qed.

Theorem rebuild_repr_fixed_refines_observably :
  rebuild_repr_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc.
  exact (rebuild_repr_fixed_observably_never_returns flag is_local u e T
           cannot_change dominated to_unbox tcr pdc (fun _ => false)).
Qed.

Theorem rebuild_repr_fixed_tail_refines_observably_local :
  rebuild_repr_fixed_tail_refines_observably_local_statement.
Proof. exact rebuild_repr_fixed_refines_observably_local. Qed.

Theorem rebuild_repr_fixed_tail_refines_observably :
  rebuild_repr_fixed_tail_refines_observably_statement.
Proof. exact rebuild_repr_fixed_refines_observably. Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

Lemma no_local_slots : is_local_slots_only no_local.
Proof. intros f H. discriminate H. Qed.

Theorem block_in_slot_example : block_in_slot_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  apply (rebuild_repr_fixed_refines_observably Auto no_local interface
           block_in_slot block_in_slot_tables []);
    try (vm_compute; reflexivity).
  - exact no_local_slots.
  - left. intros f. reflexivity.
Qed.

Theorem two_fields_example : two_fields_example_statement.
Proof.
  assert (Heq : rebuild_unit_unbox_repr_fixed no_local TwoFieldsExample.facts
                  TwoFieldsExample.tables [] TwoFieldsExample.code_deps
                  TwoFieldsExample.dominated TwoFieldsExample.to_unbox
                  TwoFieldsExample.tcr true TwoFieldsExample.unit_expr
                = TwoFieldsExample.rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Heq | ].
  split; [ vm_compute; reflexivity | ].
  rewrite <- Heq.
  apply (rebuild_repr_fixed_refines_observably Auto no_local interface
           TwoFieldsExample.unit_expr TwoFieldsExample.tables []);
    try (vm_compute; reflexivity).
  - exact no_local_slots.
  - left. intros f. reflexivity.
Qed.

Theorem repr_never_returns_example : repr_never_returns_example_statement.
Proof.
  assert (Heq : rebuild_unit_unbox_repr_fixed no_local ReprNeverExample.facts
                  ReprNeverExample.tables [] ReprNeverExample.code_deps
                  ReprNeverExample.dominated ReprNeverExample.to_unbox
                  ReprNeverExample.tcr true ReprNeverExample.unit_expr
                = ReprNeverExample.rebuilt)
    by (vm_compute; reflexivity).
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Heq | ].
  rewrite <- Heq.
  apply (rebuild_repr_fixed_observably_never_returns Auto no_local interface
           ReprNeverExample.unit_expr ReprNeverExample.tables []
           ReprNeverExample.dominated ReprNeverExample.to_unbox
           ReprNeverExample.tcr true ReprNeverExample.nr);
    try (vm_compute; reflexivity).
  - exact no_local_slots.
  - left. intros f. reflexivity.
Qed.

Theorem unboxed_closure_goes_wrong : unboxed_closure_goes_wrong_statement.
Proof.
  assert (Hsrc : run 10 (on_its_own unboxed_closure)
                 = Exit 1%positive [Value_const (Tagged_immediate 7)]
                     empty_heap no_reads)
    by (vm_compute; reflexivity).
  assert (Htgt : run 10 (on_its_own unboxed_closure_rebuilt) = Undefined)
    by (vm_compute; reflexivity).
  split; [ exact Hsrc | split; [ exact Htgt | ] ].
  intros (_ & _ & Hwrong).
  assert (Hw : behaves (on_its_own unboxed_closure) Goes_wrong)
    by (apply Hwrong; exists 10; exact Htgt).
  assert (Hx : behaves (on_its_own unboxed_closure)
                 (Exits 1%positive [Value_const (Tagged_immediate 7)]
                    empty_heap))
    by (exists 10, no_reads; exact Hsrc).
  discriminate (behaviour_unique _ _ _ Hw Hx).
Qed.

Theorem unboxed_closure_lifted : unboxed_closure_lifted_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem static_set_example : static_set_example_statement.
Proof.
  assert (Hsrc : run 10 (on_its_own static_set)
                 = Exit 1%positive [Value_const (Tagged_immediate 7)]
                     empty_heap no_reads)
    by (vm_compute; reflexivity).
  assert (Htgt : run 10 (on_its_own static_set_rebuilt)
                 = Exit 1%positive [Value_const (Tagged_immediate 7)]
                     empty_heap no_reads)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hsrc | split; [ exact Htgt | ] ].
  apply refines_observably_of_refines.
  apply (FlambdaReaperTupledProofs.same_exit_refines _ _ 1%positive
           [Value_const (Tagged_immediate 7)] empty_heap);
    [ exists 10, no_reads; exact Hsrc | exists 10, no_reads; exact Htgt ].
Qed.
