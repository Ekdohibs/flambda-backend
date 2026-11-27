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

(* Proofs of FlambdaReaperTypesReprClosuresSpec.v. This file needs no
   review.

   [crel_exact_rebuilt]: [crel_exact] of the simulation, with the kinds of
   the new slots not unboxed in turn, is [changed_closure_rebuilt]: the
   value slots of the closure of the third run are the leaves of the
   representation in their order, each once, so the value of the leaf at
   each position is that of its slot ([forall2_by_keys]).

   [reprs_dominated]: a name that has a representation is dominated by an
   allocation that has the same one ([dinv], kept by the fold of
   [changed_representations] beside
   FlambdaReaperTypesTraversalProofs.rinv: the allocation that dominates
   a name of the set of [x] has no representation before [x] is met, and
   gets that of its own function slot in the set of [x]). With
   [check_dominated], the allocations of the name are that allocation
   alone, so [changed_at] holds ([changed_at_of]).

   [rr_blocks]: the relation [rr] binds static blocks only
   ([rr_bindings]), so the static environment of the first run of [rr]
   (the unit rebuilt with changes of calling convention) binds static
   blocks only, and [static_sets_whole] holds with nothing asked
   ([sets_whole_of_blocks]).

   [repr_runs]: the three runs, as FlambdaReaperTypesCallsProofs.v reads
   them (FlambdaReaperTypesCallsProofs.cc_states), the third related to
   the second by [rr] with its names those of the unit
   (FlambdaReaperUnboxingRefinesClosuresProofs
   .rebuild_repr_reach_modes_with_rr_names), its run well-kinded
   (FlambdaWellKindedProofs.run_well_kinded). The statements are then
   FlambdaReaperTypesTraversalUnaliasedProofs.changed_type_sound_full and
   [stored_type_sound_full] applied to them. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperTypes FlambdaReaperTypesSpec
  FlambdaReaperTypesTermSpec FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesReprClosuresSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingSpec
  FlambdaReaperUnboxingProofs FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSimProofs
  FlambdaReaperUnboxingRefinesClosuresProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesTraversalSpec FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesTraversalUnaliasedSpec
  FlambdaReaperTypesTraversalUnaliasedProofs FlambdaReaperTypesCallsProofs
  FlambdaReaperTypesCallsExitProofs FlambdaReaperTypesReprExitProofs
  FlambdaReaperFlowProofs FlambdaReaperUnboxingFuelProofs
  FlambdaWellKindedSpec FlambdaWellKindedProofs FlambdaRenameApartSpec
  FlambdaReaperCallingConventionsProofs FlambdaReaperLeastProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypeMeaningBoolProofs
  FlambdaTypingEnvProofs FlambdaEnvAliasesProofs FlambdaTypeGrammarProofs.
Import ListNotations.

Module SIM := FlambdaReaperUnboxingRefinesClosuresSimProofs.
Module CP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UCS := FlambdaReaperUnboxingRefinesCallsSimProofs.
Module TP := FlambdaReaperTypesProofs.
Module TT := FlambdaReaperTypesTraversalProofs.
Module TTS := FlambdaReaperTypesTraversalSpec.
Module UTP := FlambdaReaperTypesTraversalUnaliasedProofs.
Module TCP := FlambdaReaperTypesCallsProofs.
Module RX := FlambdaReaperTypesReprExitProofs.
Module CX := FlambdaReaperTypesCallsExitProofs.
Module WK := FlambdaWellKindedSpec.
Module WKP := FlambdaWellKindedProofs.

(* ======================================================================== *)
(* The closure of the third run                                             *)
(* ======================================================================== *)

Lemma value_at_path_eq (p : list field) (w : value) :
  value_at_path p w = URP.path_value p w.
Proof.
  revert w. induction p as [ | f p IH]; intros w; [ reflexivity | ].
  cbn [value_at_path URP.path_value].
  replace (field_of_value f w) with (URP.field_value f w)
    by (destruct f, w; reflexivity).
  destruct (URP.field_value f w); [ apply IH | reflexivity ].
Qed.

Lemma lookup_of_in {A : Type} (l : list (positive * A)) (k : positive)
    (a : A) :
  (k, a) ∈ l -> NoDup (map fst l) -> lookup k l = Some a.
Proof.
  induction l as [ | [k0 a0] l IH]; intros Hin Hnd; [ destruct Hin | ].
  cbn [map fst] in Hnd. inversion Hnd as [ | k1 l1 Hk0 Hnd' [Heq1 Heq2]].
  subst. destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. apply lookup_head.
  - rewrite lookup_other; [ exact (IH Hin Hnd') | ].
    intros Hk. apply Hk0. rewrite <- Hk. apply in_map_iff. exists (k, a).
    split; [ reflexivity | exact Hin ].
Qed.

(* A list of values whose keys are those of [leaves], related position by
   position to [paths] as [Q] says of each key and its value. *)
Lemma forall2_by_keys {B : Type} (leaves : list (positive * flambda_kind))
    (paths : list B) (P : positive * flambda_kind -> B -> Prop)
    (Q : B -> value -> Prop) (vs0 : list (positive * value)) :
  Forall2 P leaves paths -> map fst vs0 = map fst leaves ->
  (∀ (vk : positive * flambda_kind) (pk : B) (x' : value),
     P vk pk -> (fst vk, x') ∈ vs0 -> Q pk x') ->
  Forall2 Q paths (map snd vs0).
Proof.
  intros H. revert vs0.
  induction H as [ | vk pk leaves paths Hp _ IH]; intros vs0 Hk HQ.
  - destruct vs0; [ constructor | discriminate Hk ].
  - destruct vs0 as [ | [s x'] vs0]; [ discriminate Hk | ].
    cbn [map] in Hk |- *. injection Hk as Hs Hk. cbn [fst snd] in Hs.
    constructor.
    + apply (HQ vk pk x' Hp). left. rewrite <- Hs. reflexivity.
    + apply IH; [ exact Hk | ].
      intros vk' pk' x'' Hp' Hin. apply (HQ vk' pk' x'' Hp'). right.
      exact Hin.
Qed.

(* [crel_exact], with the kinds of the new slots not unboxed in turn, is
   [changed_closure_rebuilt]. *)
Lemma crel_exact_rebuilt (r : U.closure_repr) (v v' : value) :
  SIM.crel_exact r v v' ->
  (∀ (decls : list (function_slot * code_id))
     (vs' : list (value_slot * value)) (w s : value_slot) (x' : value),
     v'
     = Value_closure_of_set (U.current_slot r)
         (U.rename_function_decls r decls) vs' ->
     (Field_value_slot w, U.Not_unboxed s) ∈ U.cr_fields r ->
     lookup s vs' = Some x' -> kind_of_value x' = Value) ->
  changed_closure_rebuilt r v v'.
Proof.
  intros [(decls & vs & vs' & -> & -> & Hdecls & Hslots)
          (decls2 & vs2 & vs2' & Heq1 & Heq2 & _ & Hkeys & Hnd & Hvsf)]
    Hkind.
  injection Heq1 as <- <-. injection Heq2 as <-.
  exists decls, vs, vs'.
  split; [ reflexivity | split; [ reflexivity | ] ].
  split; [ exact Hdecls | split; [ exact Hkeys | ] ].
  apply (forall2_by_keys _ _ _ _ vs'
           (UCS.fields_leaves_paths (U.cr_fields r)) Hkeys).
  intros [s k] [p k'] x' [Hk Hp] Hin. cbn [fst snd] in Hk, Hp, Hin |- *.
  subst k'.
  pose proof (lookup_of_in vs' s x' Hin Hnd) as Hl.
  destruct (URP.upath_unboxed_inv _ _ _ _ _ Hp)
    as (f & u & p0 & -> & Hfu & Hp0).
  destruct (Hvsf _ Hfu) as (w & u' & Heq). injection Heq as -> <-.
  destruct (Hslots w u Hfu) as (v0 & Hw & Hh).
  cbn [slot_value_at_path]. rewrite Hw.
  destruct u as [s0 | nested]; cbn [SIM.slot_holds] in Hh.
  - destruct (URP.upath_not_unboxed_inv _ _ _ _ _ Hp0) as (-> & -> & ->).
    rewrite Hl in Hh. injection Hh as <-.
    split; [ reflexivity | ].
    exact (Hkind decls vs' w s0 x' eq_refl Hfu Hl).
  - destruct (Hh p0 s k Hp0) as (w' & Hpw & Hl' & Hk').
    rewrite Hl in Hl'. injection Hl' as <-.
    rewrite value_at_path_eq. split; [ exact Hpw | exact Hk' ].
Qed.

(* A value that [rebuilt_value] relates to a closure of a set is a closure
   of that set at the same function slot. *)
Lemma rebuilt_closure_inv (G : graph) (sol : solution) (v : value)
    (n : node) (slot : function_slot) (decls : list (function_slot * code_id))
    (vs' : list (value_slot * value)) :
  rebuilt_value G sol v n (Value_closure_of_set slot decls vs') ->
  ∃ vs : list (value_slot * value), v = Value_closure_of_set slot decls vs.
Proof.
  destruct v as [c | code vs0 | slot0 fdecls vs | s | tag fields | elements
                | c | l | callee args arity];
    cbn [rebuilt_value]; intros H; try contradiction.
  - destruct H as [_ H]. discriminate H.
  - destruct H as (sites & vs1 & Heq & _). injection Heq as -> -> _.
    exists vs. reflexivity.
  - destruct H as [H _]. discriminate H.
  - destruct H as (a & fields' & _ & _ & H & _). discriminate H.
  - destruct H as (elements' & H). discriminate H.
  - destruct H as (a & bn & from & _ & _ & _ & _ & H).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate H.
  - destruct H as (closure' & args' & H). discriminate H.
Qed.

(* An allocation with a source that a closure of a set may come from. *)
Lemma closure_allocation (is_local : field -> bool) (G : graph)
    (sol : solution) (slot : function_slot)
    (fdecls : list (function_slot * code_id)) (vs : list (value_slot * value))
    (id : code_id) (n : node) (v' : value) :
  closed is_local G sol -> lookup slot fdecls = Some id ->
  rebuilt_value G sol (Value_closure_of_set slot fdecls vs) n v' ->
  ∃ a : node, may_come_from sol a n ∧ s_has_source sol a = true.
Proof.
  intros Hc Hl Hrv. cbn [rebuilt_value] in Hrv.
  destruct Hrv as (sites & vs_mid & -> & Hmcf & Hsites & _).
  assert (Hb : is_bound slot fdecls = true)
    by (unfold is_bound; rewrite Hl; reflexivity).
  exists (sites slot). split; [ exact Hmcf | ].
  apply (has_source_function_slot _ _ _ Hc _ _ (sites slot)
           (proj2 (Hsites slot id Hl slot (sites slot)) (conj Hb eq_refl))
           eq_refl).
Qed.

(* ======================================================================== *)
(* The allocation that dominates a name that changes                        *)
(* ======================================================================== *)

Section Dominated.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable cm : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.
Variable tcr : list node.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Hypothesis Hc : closed is_local G sol.
Hypothesis Hsets : TTS.function_slots_of_sets G.
Hypothesis Hmut : function_slots_mutual G.
Hypothesis Hdom : U.check_dominated T dominated = true.
Hypothesis Hone : TTS.changed_closures_one_allocation G sol tcr.

Local Abbreviation fsf a g b := (g_constructor G a (Field_function_slot g) b).

(* A name with a representation is dominated by an allocation with the
   same one. *)
Definition dinv (acc : list (node * U.closure_repr)) : Prop :=
  ∀ (c : node) (r : U.closure_repr),
    Keyed.lookup node_eqb c acc = Some r ->
    ∃ f : node,
      (f, c) ∈ dominated ∧ Keyed.lookup node_eqb f acc = Some r.

Lemma dinv_step (acc : list (node * U.closure_repr)) (seen : list node)
    (x : node) (sx : list (function_slot * node)) (fields : U.unboxed_fields)
    (n1 n2 : positive) (fss : list (function_slot * function_slot)) :
  TT.rinv F tcr acc seen -> dinv acc -> x ∈ tcr ->
  mem_node x seen = false -> U.set_function_slots F x = sx ->
  U.rename_slots (map fst sx) n1 = (fss, n2) ->
  dinv
    (acc
     ++ flat_map
          (fun '(fs, f) =>
             flat_map
               (fun '(a, c) =>
                  if node_eqb a f
                  then [(c, U.Closure_repr fields fss fs)]
                  else [])
               dominated)
          sx).
Proof.
  intros [Hinv1 _] Hd Hx Hseen Hsx _ c r Hl. subst sx.
  set (sx := U.set_function_slots F x) in *.
  set (new := flat_map _ sx) in *.
  rewrite keyed_lookup_app in Hl.
  destruct (Keyed.lookup node_eqb c acc) as [r0 | ] eqn:Hc0.
  - injection Hl as <-. destruct (Hd c r0 Hc0) as (f & Hf & Hlf).
    exists f. split; [ exact Hf | ]. rewrite keyed_lookup_app, Hlf.
    reflexivity.
  - apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hl.
    apply TT.in_new in Hl as (fs & f & Hfs & Hdcf & ->).
    exists f. split; [ exact Hdcf | ].
    pose proof (proj1 (TT.sfs_iff F Hsets x f fs) Hfs) as Hxf.
    destruct (Hmut x f fs Hxf) as (g' & Hfx).
    rewrite keyed_lookup_app.
    destruct (Keyed.lookup node_eqb f acc) as [r0 | ] eqn:Hf0.
    + exfalso.
      destruct (Hinv1 f r0 g' x Hfx Hf0) as (x0 & _ & Hcur0 & _ & Hall0).
      pose proof (proj1 (TT.sfs_iff F Hsets x0 f _) Hcur0) as Hx0f.
      pose proof (proj2 ((proj2 (proj2 Hsets)) x0 f x _ g' Hx0f) Hfx)
        as Hx0x.
      destruct (Hall0 g' x (proj2 (TT.sfs_iff F Hsets x0 x g') Hx0x))
        as [_ Hm].
      rewrite Hm in Hseen. discriminate Hseen.
    + assert (Hff : (f, f) ∈ dominated)
        by exact (TT.self_dominated is_local F T dominated tcr Hc Hmut Hdom
                    Hone x f fs Hx Hxf).
      assert (Hinn : (f, U.Closure_repr fields fss fs) ∈ new)
        by (apply TT.in_new; exists fs, f; auto).
      destruct (TT.keyed_lookup_some f _ new Hinn) as (r1 & Hr1).
      rewrite Hr1. f_equal.
      apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hr1.
      apply TT.in_new in Hr1 as (fs1 & f1 & Hfs1 & Hd1 & ->).
      rewrite (TT.dominated_self is_local F T dominated Hc Hdom f1 f x g'
                 Hfx Hd1) in Hfs1.
      f_equal.
      exact ((proj1 (proj2 Hsets)) x f fs1 fs
               (proj1 (TT.sfs_iff F Hsets x f fs1) Hfs1) Hxf).
Qed.

Lemma dinv_fold :
  ∀ next : positive,
    dinv (fst (U.changed_representations F T cm dominated to_unbox
                 is_local tcr next)).
Proof.
  intros next. rewrite FlambdaReaperUnboxingFuelProofs
                         .changed_representations_depth.
  match goal with
  | |- context [List.fold_left ?h tcr _] => set (step := h)
  end.
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * U.closure_repr)) (seen : list node)
      (n : positive),
      (∀ y : node, y ∈ l -> y ∈ tcr) -> TT.rinv F tcr acc seen -> dinv acc ->
      TT.rinv F tcr (fst (fst (List.fold_left step l (acc, seen, n))))
        (snd (fst (List.fold_left step l (acc, seen, n))))
      ∧ dinv (fst (fst (List.fold_left step l (acc, seen, n))))).
  { induction l as [ | x l IH]; intros acc seen n Hl Hinv Hd;
      [ split; [ exact Hinv | exact Hd ] | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    assert (Hx : x ∈ tcr) by (apply Hl; left; reflexivity).
    assert (Hl' : ∀ y : node, y ∈ l -> y ∈ tcr)
      by (intros y Hy; apply Hl; right; exact Hy).
    unfold step in Hstep. cbv beta match in Hstep.
    destruct (mem_node x seen) eqn:Hseen;
      [ injection Hstep as <- <- _; exact (IH acc seen n' Hl' Hinv Hd) | ].
    destruct (U.set_function_slots F x) as [ | [fs0 f00] l0] eqn:Hsx;
      [ injection Hstep as <- <- _; exact (IH acc seen n' Hl' Hinv Hd) | ].
    destruct (U.mk_unboxed_fields F T cm dominated to_unbox
                (FlambdaReaperUnboxingFuelSpec.unboxing_depth T) x
                (U.get_fields_of_constructors F T is_local
                   (map snd ((fs0, f00) :: l0)))
                n)
      as [fields n1] eqn:Hmk.
    destruct (U.rename_slots (map fst ((fs0, f00) :: l0)) n1)
      as [fss n2] eqn:Hren.
    injection Hstep as <- <- _.
    apply IH; [ exact Hl' | | ].
    - exact (TT.rinv_step is_local F T dominated tcr Hc Hsets Hmut Hdom Hone
               acc seen x _ fields n1 n2 fss Hinv Hx Hseen Hsx Hren).
    - exact (dinv_step acc seen x _ fields n1 n2 fss Hinv Hd Hx Hseen Hsx
               Hren). }
  destruct (List.fold_left step tcr ([], [], next)) as [[acc seen] n] eqn:Hf.
  cbn [fst].
  assert (H0 : TT.rinv F tcr [] []).
  { split; [ intros c r g b _ H; discriminate H | intros p [] ]. }
  pose proof (Hgen tcr [] [] next (fun y H => H) H0
                ltac:(intros c r H; discriminate H)) as H.
  rewrite Hf in H. exact (proj2 H).
Qed.

End Dominated.

(* What a name whose representation changes holds, in the third run, is in
   the representation of each allocation it may come from. *)
Lemma changed_at_of (sol : solution) (dominated : list (node * node))
    (reprs : list (node * U.closure_repr)) (T : tables) (x : node)
    (r : U.closure_repr) (v_mid v' : value) :
  sol = solution_of_tables T ->
  U.check_dominated T dominated = true ->
  (∀ (c : node) (r : U.closure_repr),
     Keyed.lookup node_eqb c reprs = Some r ->
     ∃ f : node,
       (f, c) ∈ dominated ∧ Keyed.lookup node_eqb f reprs = Some r) ->
  Keyed.lookup node_eqb x reprs = Some r ->
  (∃ a : node, may_come_from sol a x ∧ s_has_source sol a = true) ->
  changed_closure_rebuilt r v_mid v' ->
  changed_at sol reprs x v_mid v'.
Proof.
  intros -> Hdom Hdinv Hx Hex Hcr. split; [ exact Hex | ].
  intros a Ha Hha.
  destruct (Hdinv x r Hx) as (f & Hfx & Hlf).
  destruct (proj1 (FlambdaReaperUnboxingProofs.check_dominated_sound T
                     dominated Hdom f x)
              (proj2 (mem_pair_In f x dominated) Hfx))
    as (Hsrc & Hhs & Hnm).
  destruct Ha as (_ & [Has | Hsa] & _).
  { exfalso. apply Hnm. left. exact Has. }
  destruct (node_eq_dec a f) as [-> | Hne].
  - exists r. split; [ exact Hlf | exact Hcr ].
  - exfalso. apply Hnm. right. exists a, f. auto 6.
Qed.

(* ======================================================================== *)
(* The static environment of the first run of [rr]                          *)
(* ======================================================================== *)

(* A static environment that binds static blocks only. *)
Definition statics_blocks (S : static_env) : Prop :=
  ∀ (sym : symbol) (a : statically_allocated_value),
    lookup sym (static_values S) = Some a ->
    ∃ (tag : Z) (fields : list value), a = Static_block tag fields.

Lemma statics_blocks_empty : statics_blocks empty_static_env.
Proof. intros sym a H. discriminate H. Qed.

(* Only a [Let_static] changes the static environment. *)
Lemma step_statics_same (s s1 : state) :
  (∀ (bs : list static_binding) (body : expr),
     state_expr s <> Let_static bs body) ->
  step s = Step s1 -> state_statics s1 = state_statics s.
Proof.
  destruct s as [e vars conts statics hp rd]. cbn [state_expr state_statics].
  intros Hns.
  destruct e as [x kind d body | bindings body | k h body | handlers body
                | ap | ac | scrutinee arms | ].
  - rewrite step_let_exec.
    destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb _ _); [ | discriminate ].
    intros [= <-]. reflexivity.
  - exfalso. exact (Hns bindings body eq_refl).
  - rewrite step_let_cont_non_recursive. intros [= <-]. reflexivity.
  - rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. reflexivity.
  - rewrite step_apply.
    destruct (eval_simple _ (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples _ (ap_args ap)); [ | discriminate ].
    destruct (eval_simples _ (exn_extra_simples ap)); [ | discriminate ].
    intros Hstep.
    exact (FlambdaReaperFlowProofs.call_function_statics _ _ _ _ _ _ _ _ _ _
             _ Hstep).
  - rewrite step_apply_cont.
    destruct (eval_simples _ (ac_args ac)) as [vs | ]; [ | discriminate ].
    intros Hstep. unfold apply_continuation in Hstep.
    exact (FlambdaReaperFlowProofs.apply_cont_value_statics _ _ _ _ _ _
             Hstep).
  - rewrite step_switch.
    destruct (eval_simple _ scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. reflexivity.
  - rewrite step_invalid. discriminate.
Qed.

Section Rr_blocks.

Variable M : list (node * U.unboxed_fields).
Hypothesis M_distinct : distinct (URP.field_variables M) = true.
Variable CH : variable -> option U.closure_repr.
Hypothesis CH_M : ∀ (x : variable) (r : U.closure_repr),
  CH x = Some r -> FlambdaReaperUnboxingRefinesSpec.fields_of M x = Some [].
Hypothesis CH_fresh : ∀ (x : variable) (r : U.closure_repr),
  CH x = Some r -> among x (URP.field_variables M) = false.
Variable CS : code_id -> option SIM.code_info.
Variable RCH : bool -> variable -> code_id -> Prop.

(* The static bindings that [rr] relates bind static blocks and code. *)
Lemma eval_bindings_blocks (bs bs' : list static_binding) :
  SIM.rr_bindings M CH CS RCH bs bs' ->
  ∀ (vars : env) (S S' : static_env),
    statics_blocks S -> eval_static_bindings vars bs S = Some S' ->
    statics_blocks S'.
Proof.
  induction 1 as [ | b bs bs' Hb _ IH | id bs bs' _ IH
                 | id c c' ci bs bs' _ _ _ IH];
    intros vars S S' HS He; cbn [eval_static_bindings] in He.
  - injection He as <-. exact HS.
  - destruct b as [id c | id | sym const]; try discriminate Hb.
    destruct const; try discriminate Hb.
    destruct (lookup sym (static_values S)); [ discriminate He | ].
    cbn [eval_static_const] in He.
    destruct (eval_simples vars _) as [vs | ]; [ | discriminate He ].
    cbn in He. refine (IH vars _ S' _ He).
    intros sym' a Hl. cbn [static_values] in Hl. rewrite lookup_cons in Hl.
    destruct (Pos.eqb sym' sym).
    + injection Hl as <-. eexists _, _. reflexivity.
    + exact (HS sym' a Hl).
  - exact (IH vars S S' HS He).
  - destruct (lookup id (static_code S)); [ discriminate He | ].
    refine (IH vars _ S' _ He). intros sym a Hl.
    exact (HS sym a Hl).
Qed.

(* The first run of [rr] keeps a static environment of blocks. *)
Lemma step_blocks (s t s1 : state) :
  SIM.related M CH CS RCH s t -> statics_blocks (state_statics s) ->
  step s = Step s1 -> statics_blocks (state_statics s1).
Proof.
  intros (D & Hrr & _) HS Hstep.
  destruct (state_expr s) as [ | bs body | | | | | | ] eqn:He;
    try (rewrite (step_statics_same s s1
                    ltac:(intros bs0 body0; rewrite He; discriminate) Hstep);
         exact HS).
  inversion Hrr; subst.
  match goal with
  | H : SIM.rr_bindings _ _ _ _ bs ?bs' |- _ => rename H into Hbs
  end.
  rewrite (state_eta s), He, step_let_static in Hstep.
  destruct (eval_static_bindings _ bs _) as [S' | ] eqn:Hev;
    [ | discriminate Hstep ].
  injection Hstep as <-. cbn [state_statics].
  exact (eval_bindings_blocks bs _ Hbs _ _ _ HS Hev).
Qed.

(* [rr], with the invariant of the flow, and the static environment of
   the first run of blocks. *)
Definition rr_blocks (s t : state) : Prop :=
  RX.rr_rel M CH CS RCH s t ∧ statics_blocks (state_statics s).

Lemma rr_blocks_answered (s t : state) :
  rr_blocks s t -> answered rr_blocks eq measure s t.
Proof.
  intros [Hr HS].
  pose proof (RX.rr_rel_answered M M_distinct CH CH_M CH_fresh CS RCH s t Hr)
    as Ha.
  unfold answered in Ha |- *.
  destruct (step s) as [s1 | k vs hp rd1 | ] eqn:Hs; [ | exact Ha | exact I ].
  pose proof (step_blocks s t s1 (proj1 Hr) HS Hs) as HS1.
  destruct Ha
    as [H | [(t1 & Ht & Hr1) | [(Hlt & Hr1) | (t1 & n & Ht & Hr1)]]].
  - left. exact H.
  - right. left. exists t1. exact (conj Ht (conj Hr1 HS1)).
  - right. right. left. exact (conj Hlt (conj Hr1 HS1)).
  - right. right. right. exists t1, n. exact (conj Ht (conj Hr1 HS1)).
Qed.

(* Each state that the first run comes to, and from which it steps, is
   related to one that the second comes to, and binds static blocks
   only. *)
Lemma rr_reach_blocks (p q : expr) :
  SIM.rr M CH CS RCH [] p q ->
  SIM.flow_run RCH (State p [] [] empty_static_env empty_heap no_reads) ->
  ∀ (n : nat) (t : state),
    run n (State p [] [] empty_static_env empty_heap no_reads) = Step t ->
    step t <> Undefined ->
    ∃ (m : nat) (t' : state),
      run m (State q [] [] empty_static_env empty_heap no_reads) = Step t'
      ∧ SIM.related M CH CS RCH t t' ∧ statics_blocks (state_statics t).
Proof.
  intros Hrr Hfl n t Hrun Ht.
  destruct (CX.shrinking_reaches rr_blocks eq measure rr_blocks_answered n
              _ _ t (conj (RX.rr_initial M CH CS RCH p q Hrr Hfl)
                       statics_blocks_empty)
              Hrun Ht)
    as (m & t' & Hm & [[Hr _] HS]).
  exists m, t'. split; [ exact Hm | split; [ exact Hr | exact HS ] ].
Qed.

End Rr_blocks.

(* With the static environment of the second run of blocks,
   [static_sets_whole] asks nothing. *)
Lemma sets_whole_of_blocks (G : graph) (sol : solution)
    (statics statics2 statics3 : static_env) :
  statics_rebuilt G sol statics statics2 -> statics_blocks statics2 ->
  static_sets_whole sol statics statics3.
Proof.
  intros Hst HS sym slot csyms fdecls vs Hl Hhu.
  pose proof (Hst sym _ Hl Hhu) as H. cbn [static_rebuilt] in H.
  destruct H as (sites & vs' & Hf & _).
  destruct (HS sym _ Hf) as (tag & fields & Heq). discriminate Heq.
Qed.

(* ======================================================================== *)
(* The three runs                                                           *)
(* ======================================================================== *)

(* The value slots of a closure that is well-kinded hold values of the
   kinds that the table gives them. *)
Lemma well_kinded_slot (slots : WK.slot_kinds) (slot : function_slot)
    (decls : list (function_slot * code_id)) (vs : list (value_slot * value))
    (s : value_slot) (x : value) :
  WK.value_well_kinded slots (Value_closure_of_set slot decls vs) = true ->
  lookup s vs = Some x -> lookup s slots = Some Value ->
  kind_of_value x = Value.
Proof.
  intros Hwk Hl Hs. simpl in Hwk.
  pose proof (proj1 (forallb_forall _ _) Hwk (s, x) (lookup_in s x vs Hl))
    as H.
  cbn beta iota in H. apply andb_true_iff in H as [H _].
  rewrite Hs in H. cbn [WK.has_kind] in H.
  symmetry. exact (proj1 (kind_eqb_eq _ _) H).
Qed.

(* The values of a state of a well-kinded run are well-kinded. *)
Lemma well_kinded_var (slots : WK.slot_kinds) (outer : WK.arity_env)
    (t : state) (x : variable) (v : value) :
  WK.state_well_kinded slots outer t = true ->
  lookup x (state_vars t) = Some v -> WK.value_well_kinded slots v = true.
Proof.
  intros Ht Hl. unfold WK.state_well_kinded in Ht.
  apply andb_true_iff in Ht as [Ht _]. apply andb_true_iff in Ht as [Ht _].
  apply andb_true_iff in Ht as [Ht _]. apply andb_true_iff in Ht as [_ Ht].
  unfold WK.env_well_kinded in Ht.
  exact (proj1 (forallb_forall _ _) Ht (x, v) (lookup_in x v _ Hl)).
Qed.

Section Runs.

Variable is_local : field -> bool.
Variable u : unit_interface.
Variable e : expr.
Variable T : tables.
Variable cannot_change : list node.
Variable dominated : list (node * node).
Variable to_unbox tcr : list node.
Variable pdc : bool.
Variable nr : code_id -> bool.
Variable sets : list set_of_closures.
Variable slots : WK.slot_kinds.

Local Abbreviation F := (traverse_unit Auto u e).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).
Local Abbreviation reprs := (unit_reprs is_local u e T dominated to_unbox tcr).
Local Abbreviation unboxed :=
  (fst (U.unit_unboxed_fields F T dominated to_unbox e)).

Hypothesis Hprem : unit_premises false Auto is_local u e T sets.
Hypothesis Hrepr :
  repr_premises Auto is_local u e T cannot_change dominated to_unbox tcr nr.
Hypothesis Hkinds :
  repr_kinds_premises is_local u e T cannot_change dominated to_unbox tcr pdc
    slots.

(* The three runs, at a state of the first that binds [x]: the second
   binds [x] to [v_mid], related by [rebuilt_value]; the third, if [x]
   changes, to the closure of [v_mid] in the representation of [x], and,
   if [x] is neither changed nor unboxed, to [v_mid]. *)
Lemma repr_runs (n : nat) (s : state) (x : variable) (v : value) :
  run n (initial_state e) = Step s ->
  ~ behaves (initial_state e) Goes_wrong ->
  lookup x (state_vars s) = Some v ->
  s_has_usage sol (Node_variable x) = true ->
  x ∈ FlambdaRenameApartSpec.variables_of e ->
  ∃ (m2 : nat) (t2 : state) (v_mid : value) (m3 : nat) (t3 : state),
    run m2
      (initial_state
         (rebuild_unit_cc_from_tables F T cannot_change
            (unit_code_deps Auto u e) pdc e))
    = Step t2
    ∧ lookup x (state_vars t2) = Some v_mid
    ∧ rebuilt_value G sol v (Node_variable x) v_mid
    ∧ run m3
        (initial_state
           (repr_rebuilt Auto is_local u e T cannot_change dominated to_unbox
              tcr pdc))
      = Step t3
    ∧ statics_rebuilt G sol (state_statics s) (state_statics t3)
    ∧ statics_allocated sol (state_statics s)
    ∧ static_sets_whole sol (state_statics s) (state_statics t3)
    ∧ (∀ r : U.closure_repr,
         Keyed.lookup node_eqb (Node_variable x) reprs = Some r ->
         ∃ v' : value,
           lookup x (state_vars t3) = Some v'
           ∧ changed_closure_rebuilt r v_mid v'
           ∧ closure_in_set (state_statics s) v <> None
           ∧ ∃ a : node,
               may_come_from sol a (Node_variable x)
               ∧ s_has_source sol a = true)
    ∧ (Keyed.lookup node_eqb (Node_variable x) reprs = None ->
       Keyed.lookup node_eqb (Node_variable x) unboxed = None ->
       lookup x (state_vars t3) = Some v_mid).
Proof.
  intros Hrun Hng Hx Hhu Hin.
  pose proof Hprem as (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits
                       & Honce & Hvals & Hsets).
  pose proof Hrepr as (Hccc & Hwid & Hcalls & Hreads).
  destruct (TCP.least_with_checks false Auto is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  pose proof (TP.least_closed_tables is_local F T Hle) as Hc.
  destruct (TCP.cc_checks Auto u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hcc).
  assert (Hs : step s <> Undefined).
  { intros Hu. apply Hng. exists (S n).
    replace (S n) with (n + 1) by lia.
    rewrite run_add, Hrun, run_succ, Hu. reflexivity. }
  destruct (TCP.cc_states Auto is_local u e T _ ccw (unit_code_deps Auto u e)
              pdc Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact
              Honce Hvals Hcl Hw Hcc (fun _ _ => eq_refl) n s Hrun Hs)
    as (m2 & t2 & Hm2 & Hst & Halloc & Hvars & _ & _).
  assert (Hm2' :
    run m2
      (initial_state
         (rebuild_unit_cc_from_tables F T cannot_change
            (unit_code_deps Auto u e) pdc e))
    = Step t2).
  { unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
    unfold initial_state, rebuild_unit_cc_with in Hm2 |- *. exact Hm2. }
  assert (Ht2 : step t2 <> Undefined).
  { intros Hu. apply Hng.
    destruct (FlambdaReaperCallingConventionsProofs
              .rebuild_cc_refines_observably_local Auto is_local u e T
                cannot_change pdc Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
                Hcheck_exact Hccc Hwid)
      as (_ & _ & Hgw).
    apply Hgw. exists (S m2).
    replace (S m2) with (m2 + 1) by lia.
    unfold initial_state in Hm2'.
    rewrite run_add, Hm2', run_succ, Hu. reflexivity. }
  destruct (Hvars x v Hx Hhu) as (v_mid & Hx2 & Hrv).
  destruct (CP.rebuild_repr_reach_modes_with_rr_names None Auto is_local u e
              T cannot_change dominated to_unbox tcr pdc nr
              (fun _ _ _ => true) Hlocal Hfrag Hcheck
              (CP.repr_calls_within_reach_unit nr is_local _ T cannot_change
                 _ dominated to_unbox tcr (fun _ _ _ => true) e Hcalls)
              Hreads)
    as (CS & Hdist & Hrr).
  destruct (URP.unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [_ Hfresh].
  destruct (rr_reach_blocks _ Hdist _ (CP.CHf_M _ _) (CP.CHf_fresh _ _) CS
              _ _ _ Hrr
              ltac:(apply SIM.flow_run_total; intros b f id; reflexivity)
              m2 t2 Hm2 Ht2)
    as (m3 & t3 & Hm3 & Hrel & HS2).
  pose proof (RX.rr_values _ _ _ _ _ _ Hrel) as Hv3.
  destruct Hrel as (D & _ & Henv & _).
  exists m2, t2, v_mid, m3, t3.
  split; [ exact Hm2' | split; [ exact Hx2 | split; [ exact Hrv | ] ] ].
  split; [ exact Hm3 | ].
  split; [ exact (TCP.statics_rebuilt_values _ _ _ _ _ Hv3 Hst) | ].
  split; [ exact Halloc | ].
  split; [ exact (sets_whole_of_blocks G sol _ _ _ Hst HS2) | ].
  split.
  - intros r Hr.
    assert (Hch : CP.CHf unboxed reprs x = Some r).
    { rewrite (CP.CHf_var _ _ x (Hfresh x Hin)). exact Hr. }
    destruct (proj2 Henv x r v_mid Hch Hx2) as (v' & Hx3 & Hcrel).
    exists v'. split; [ exact Hx3 | ].
    destruct Hkinds as [Hwk Hvals3].
    pose proof (WKP.run_well_kinded slots _ m3 _ t3
                  (WKP.well_kinded_unit_state slots _ _ _ Hwk) Hm3)
      as Hwk3.
    split.
    + apply (crel_exact_rebuilt r v_mid v' Hcrel).
      intros decls vs' w s0 x' -> Hfu Hl.
      apply (well_kinded_slot slots (U.current_slot r)
               (U.rename_function_decls r decls) vs' s0 x'
               (well_kinded_var slots _ t3 x _ Hwk3 Hx3) Hl).
      exact (Hvals3 (Node_variable x) r w s0 Hr Hfu).
    + destruct Hcrel as [_ (decls & vs & vs' & -> & _ & Hcur & _)].
      destruct (rebuilt_closure_inv G sol v _ _ _ _ Hrv) as (vs1 & ->).
      destruct (lookup (U.cr_current r) decls) as [id | ] eqn:Hl;
        [ | exfalso; exact (Hcur eq_refl) ].
      split.
      * cbn [closure_in_set]. rewrite Hl. discriminate.
      * exact (closure_allocation is_local G sol _ _ _ id _ _ Hc Hl Hrv).
  - intros Hnone Hnu.
    assert (Hplain : URP.plain_variable (CP.M' unboxed reprs) x = true).
    { rewrite CP.plain_variable_M'. unfold URP.plain_variable.
      unfold Keyed.is_bound. rewrite Hnu, (Hfresh x Hin).
      unfold U.changed_of_node. rewrite Hnone. reflexivity. }
    rewrite <- (proj1 (proj1 Henv) x Hplain). exact Hx2.
Qed.

End Runs.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The names that change are dominated by an allocation that changes alike,
   for the graph of the traversal. *)
Lemma unit_dinv (is_local : field -> bool) (u : unit_interface) (e : expr)
    (T : tables) (dominated : list (node * node)) (to_unbox tcr : list node) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  least_solution is_local (traverse_unit Auto u e) T ->
  U.check_dominated T dominated = true ->
  dinv dominated (unit_reprs is_local u e T dominated to_unbox tcr).
Proof.
  intros Hfrag Hwf Honce Hleast Hdom.
  pose proof (TP.least_closed_tables is_local _ T (or_introl Hleast)) as Hc.
  pose proof (TT.function_slots_of_sets_traversal u e Hfrag Hwf Honce)
    as Hsets.
  pose proof (TT.function_slots_mutual_traversal u e Hfrag Hwf Honce)
    as Hmut.
  pose proof (TT.changed_closures_one_allocation_unaliased is_local _ T tcr
                Hleast Hsets Hmut
                (UTP.closures_unaliased_traversal u e Hfrag Hwf Honce))
    as Hone.
  unfold unit_reprs, U.unit_changed_representations.
  exact (dinv_fold is_local _ T _ dominated to_unbox tcr Hc Hsets Hmut Hdom
           Hone _).
Qed.

Lemma repr_changed_gen (fx : fixes) :
  FlambdaReaperTypesTraversalUnaliasedSpec.changed_type_sound_full_with fx ->
  repr_changed_types_end_to_end_with fx.
Proof.
  intros Hchanged is_local u e T cannot_change dominated to_unbox tcr pdc nr
    sets slots tenv canon valuation valuation' fuel m t n s x v r Hprem Hrepr
    (Hdom & Hunst) Hkinds
    (Hrun & Hng & Hx & Hhu & Hin & Henv & Hfr & Hcopies & Ht & Hmd & Hsat)
    Hr.
  pose proof Hprem as (Hlocal & [Hleast _] & Hfrag & _ & _ & Hwf & _ & Honce
                       & _ & Hsets).
  pose proof Hrepr as (_ & _ & Hcalls & _).
  unfold FlambdaReaperUnboxingRefinesClosuresCallsSpec.repr_calls_unboxing_nr
    in Hcalls.
  apply andb_true_iff in Hcalls as [_ Huses].
  pose proof (TP.least_closed_tables is_local _ T (or_introl Hleast)) as Hc.
  destruct (repr_runs is_local u e T cannot_change dominated to_unbox tcr pdc
              nr sets slots Hprem Hrepr Hkinds n s x v Hrun Hng Hx Hhu Hin)
    as (m2 & t2 & v_mid & m3 & t3 & Hm2 & Hx2 & Hrv & Hm3 & Hst3 & Halloc
        & Hwhole & Hch & _).
  destruct (Hch r Hr) as (v' & Hx3 & Hcr & Hcis & Hex).
  exists m2, t2, v_mid, m3, t3, v'.
  split; [ exact Hm2 | split; [ exact Hx2 | split; [ exact Hrv | ] ] ].
  split; [ exact Hm3 | split; [ exact Hx3 | split; [ exact Hcr | ] ] ].
  exact (Hchanged is_local u T sets dominated to_unbox tcr e tenv canon
           (state_statics s) (state_statics t3) valuation valuation' fuel m t
           (Node_variable x) v v_mid v' Hfrag Hwf Honce Hleast Hlocal Hsets
           Henv Hfr Halloc Hst3 Hwhole Hcopies Hdom Huses Hunst Ht Hcis Hrv
           Hmd
           (changed_at_of _ dominated _ T (Node_variable x) r v_mid v' eq_refl
              Hdom (unit_dinv is_local u e T dominated to_unbox tcr Hfrag Hwf
                      Honce Hleast Hdom)
              Hr Hex Hcr)
           Hsat).
Qed.

Theorem repr_changed_types_end_to_end :
  repr_changed_types_end_to_end_statement.
Proof. exact (repr_changed_gen all_fixes UTP.changed_type_sound_full). Qed.

Theorem repr_changed_types_end_to_end_landed :
  repr_changed_types_end_to_end_landed_statement.
Proof.
  exact (repr_changed_gen landed_fixes UTP.changed_type_sound_full_landed).
Qed.

Lemma repr_stored_gen (fx : fixes) :
  FlambdaReaperTypesTraversalUnaliasedSpec.stored_type_sound_full_with fx ->
  repr_stored_types_end_to_end_with fx.
Proof.
  intros Hstored is_local u e T cannot_change dominated to_unbox tcr pdc nr
    sets slots tenv canon valuation valuation' fuel m t n s x v Hprem Hrepr
    (Hdom & Hunst) Hkinds
    (Hrun & Hng & Hx & Hhu & Hin & Henv & Hfr & Hcopies & Ht & Hmd & Hsat)
    Hcase.
  pose proof Hprem as (Hlocal & [Hleast _] & Hfrag & _ & _ & Hwf & _ & Honce
                       & _ & Hsets).
  pose proof Hrepr as (_ & _ & Hcalls & _).
  unfold FlambdaReaperUnboxingRefinesClosuresCallsSpec.repr_calls_unboxing_nr
    in Hcalls.
  apply andb_true_iff in Hcalls as [_ Huses].
  pose proof (TP.least_closed_tables is_local _ T (or_introl Hleast)) as Hc.
  destruct (repr_runs is_local u e T cannot_change dominated to_unbox tcr pdc
              nr sets slots Hprem Hrepr Hkinds n s x v Hrun Hng Hx Hhu Hin)
    as (m2 & t2 & v_mid & m3 & t3 & Hm2 & Hx2 & Hrv & Hm3 & Hst3 & Halloc
        & Hwhole & Hch & Hpl).
  assert (Hrc :
    ∃ v' : value,
      lookup x (state_vars t3) = Some v'
      ∧ rebuilt_changed (graph_of_facts (traverse_unit Auto u e))
          (solution_of_tables T)
          (unit_reprs is_local u e T dominated to_unbox tcr) v
          (Node_variable x) v').
  { destruct (Keyed.lookup node_eqb (Node_variable x)
                (unit_reprs is_local u e T dominated to_unbox tcr))
      as [r | ] eqn:Hr.
    - destruct (Hch r eq_refl) as (v' & Hx3 & Hcr & Hcis & Hex).
      exists v'. split; [ exact Hx3 | ].
      apply (TP.rebuilt_changed_of_runs is_local _ T _ Hc Hunst
               (state_statics s) (Node_variable x) v v_mid v' Hcis Hrv).
      exact (changed_at_of _ dominated _ T (Node_variable x) r v_mid v'
               eq_refl Hdom
               (unit_dinv is_local u e T dominated to_unbox tcr Hfrag Hwf
                  Honce Hleast Hdom)
               Hr Hex Hcr).
    - destruct Hcase as [Hcx | [Hnu Hno]]; [ exfalso; exact (Hcx eq_refl) | ].
      exists v_mid. split; [ exact (Hpl eq_refl Hnu) | ].
      apply (TP.held_rc _ T _ Hunst).
      split; [ exact Hrv | split; [ exact Hhu | exact Hno ] ]. }
  destruct Hrc as (v' & Hx3 & Hrc).
  exists m3, t3, v'. split; [ exact Hm3 | split; [ exact Hx3 | ] ].
  exact (Hstored is_local u T sets dominated to_unbox tcr e tenv canon
           (state_statics s) (state_statics t3) valuation valuation' fuel m t
           (Node_variable x) v v' Hfrag Hwf Honce Hleast Hlocal Hsets Henv
           Hfr Halloc (TP.statics_changed_of_unstored _ T _ Hunst _ _ Hst3)
           (TP.sets_whole_of_unstored T _ _ _ Hwhole) Hcopies Hdom Huses Ht
           Hrc Hmd Hsat).
Qed.

Theorem repr_stored_types_end_to_end :
  repr_stored_types_end_to_end_statement.
Proof. exact (repr_stored_gen all_fixes UTP.stored_type_sound_full). Qed.

Theorem repr_stored_types_end_to_end_landed :
  repr_stored_types_end_to_end_landed_statement.
Proof.
  exact (repr_stored_gen landed_fixes UTP.stored_type_sound_full_landed).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

(* A test of [changed_unstored] on the facts and the tables: no node of a
   representation is an allocation that may come to a node held in a field
   other than a function slot, as the sources, [any_source] and
   [any_usage] of the tables show. *)
Definition check_changed_unstored (F : facts) (T : tables)
    (reprs : list (node * U.closure_repr)) : bool :=
  let sol := solution_of_tables T in
  forallb
    (fun '(a, f, from) =>
       is_function_slot f
       || forallb
            (fun '(c, _) =>
               negb
                 (s_has_source sol c && s_has_source sol from
                  && (s_any_source sol from || s_sources sol from c)
                  && (negb (s_any_source sol c) || s_any_source sol from)
                  && (negb (s_any_usage sol from) || s_any_usage sol c)))
            reprs)
    (f_constructor F).

Lemma check_changed_unstored_sound (F : facts) (T : tables)
    (reprs : list (node * U.closure_repr)) :
  check_changed_unstored F T reprs = true ->
  changed_unstored (graph_of_facts F) (solution_of_tables T) reprs.
Proof.
  intros Hcheck a f from c Hcons Hf Hmay Hhs.
  destruct (Keyed.lookup node_eqb c reprs) as [r | ] eqn:Hl;
    [ exfalso | reflexivity ].
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hl.
  pose proof (proj1 (forallb_forall _ _) Hcheck (a, f, from) Hcons) as H.
  cbn beta iota in H.
  assert (Hfs : is_function_slot f = false)
    by (destruct f; try reflexivity; exfalso; exact (Hf _ eq_refl)).
  rewrite Hfs in H. cbn [orb] in H.
  pose proof (proj1 (forallb_forall _ _) H (c, r) Hl) as Hc.
  cbn beta iota in Hc.
  destruct Hmay as (Hhf & Hsrc & Has & Hau & _).
  rewrite Hhs, Hhf in Hc. cbn [andb] in Hc.
  assert (H1 : (s_any_source (solution_of_tables T) from
                || s_sources (solution_of_tables T) from c) = true).
  { destruct Hsrc as [H1 | H1]; rewrite H1; [ reflexivity | ].
    apply orb_true_r. }
  rewrite H1 in Hc. cbn [andb] in Hc.
  destruct (s_any_source (solution_of_tables T) c) eqn:Hac;
    [ rewrite (Has eq_refl) in Hc | ];
    destruct (s_any_usage (solution_of_tables T) from) eqn:Hauf;
    try rewrite (Hau eq_refl) in Hc; discriminate Hc.
Qed.

Ltac in_cases H :=
  repeat match type of H with
         | _ ∨ _ => destruct H as [H | H]
         end;
  try discriminate H; try contradiction H.

Module ReprTypesRuns.

Import ReprTypesExample.
Local Open Scope positive_scope.

Lemma cons_in (F : facts) (a : node) (f : field) (b : node) :
  g_constructor (graph_of_facts F) a f b -> (a, f, b) ∈ f_constructor F.
Proof. exact (fun H => H). Qed.

Lemma sets_ok :
  sets_of_graph (graph_of_facts (traverse_unit Auto U.Examples.interface
                                   unit_expr))
    ChangedExample.sets.
Proof.
  intros fdecls sites slot id Hl Hsites.
  exists (hd [] ChangedExample.sets).
  assert (Hb0 : is_bound slot fdecls = true)
    by (unfold is_bound; rewrite Hl; reflexivity).
  pose proof (proj2 (Hsites slot id Hl slot (sites slot)) (conj Hb0 eq_refl))
    as Hself.
  apply cons_in in Hself.
  change (traverse_unit Auto U.Examples.interface unit_expr)
    with U.Examples.block_in_slot_facts in Hself.
  rewrite TP.changed_example_constructors in Hself.
  cbn [In] in Hself. in_cases Hself.
  injection Hself as Hs70 E70. subst slot.
  assert (Hdom : ∀ g : function_slot,
             is_bound g fdecls = true -> g = 70%positive).
  { intros g Hg. pose proof (proj2 (Hsites 70%positive id Hl g (sites g))
                               (conj Hg eq_refl)) as Hc.
    apply cons_in in Hc.
    change (traverse_unit Auto U.Examples.interface unit_expr)
      with U.Examples.block_in_slot_facts in Hc.
    rewrite TP.changed_example_constructors, <- Hs70 in Hc.
    cbn [In] in Hc. in_cases Hc.
    injection Hc as <- _. reflexivity. }
  split; [ left; reflexivity | split ].
  - intros g c code Hin. destruct Hin as [Hin | []].
    injection Hin as <- <- _. split; [ exact Hb0 | ].
    unfold ChangedExample.x60. congruence.
  - intros g Hg. rewrite (Hdom g Hg). reflexivity.
Qed.

Lemma not_wrong : ~ behaves (initial_state unit_expr) Goes_wrong.
Proof.
  intros Hw.
  assert (Hex : behaves (initial_state unit_expr)
                  (Exits 1 [Value_const (Tagged_immediate 7)] empty_heap)).
  { exists 20%nat, no_reads. vm_compute. reflexivity. }
  discriminate (FlambdaSemanticsProofs.behaviour_unique _ _ _ Hw Hex).
Qed.

Lemma create_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics [] Typing_env.create.
Proof.
  split; [ vm_compute; reflexivity | split ].
  - intros n ty bt w Hl _. vm_compute in Hl. discriminate Hl.
  - change (Typing_env.aliases Typing_env.create) with Aliases.empty.
    apply FlambdaEnvAliasesProofs.empty_respected.
Qed.

Lemma reprs_eq :
  unit_reprs U.Examples.no_local U.Examples.interface unit_expr tables
    dominated to_unbox tcr
  = ChangedExample.reprs.
Proof. vm_compute. reflexivity. Qed.

End ReprTypesRuns.

Lemma repr_types_example_gen (fx : fixes) :
  repr_rewritten fx U.Examples.no_local U.Examples.interface
    ReprTypesExample.unit_expr ReprTypesExample.tables ChangedExample.sets
    ReprTypesExample.dominated ReprTypesExample.to_unbox
    ReprTypesExample.tcr Typing_env.create ReprTypesExample.canon 6
    (Single_source ChangedExample.x60) ChangedExample.closure_type
  = ChangedExample.rewritten ->
  repr_types_example_with fx.
Proof.
  intros Hrw.
  split.
  { split; [ intros f H; discriminate H | split ].
    { split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
               vm_compute; reflexivity | ].
      left. intros f. reflexivity. }
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; reflexivity | exact ReprTypesRuns.sets_ok ]. }
  split.
  { split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; vm_compute; reflexivity. }
  split.
  { split; [ vm_compute; reflexivity | ].
    apply check_changed_unstored_sound. vm_compute. reflexivity. }
  split.
  { split; [ vm_compute; reflexivity | ].
    intros a r w s0 Hl Hin.
    rewrite ReprTypesRuns.reprs_eq in Hl.
    apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hl.
    vm_compute in Hl.
    destruct Hl as [Hl | [Hl | [Hl | []]]]; injection Hl as _ <-;
      vm_compute in Hin; destruct Hin as [Hin | []]; discriminate Hin. }
  split.
  { split; [ unfold ReprTypesExample.state_at; vm_compute; reflexivity | ].
    split; [ exact ReprTypesRuns.not_wrong | ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; tauto | ].
    split; [ apply ReprTypesRuns.create_satisfied | ].
    split; [ intros n ty bt H; vm_compute in H; discriminate H | ].
    split.
    { intros s x k m Hcan. unfold value_of_name.
      exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
               _ _ _ _ _ _ _ _ (ReprTypesRuns.create_satisfied
                                  (state_statics ReprTypesExample.state_at))
               Hcan). }
    split.
    { split; [ vm_compute; reflexivity | ].
      intros s k w Hin Hw. vm_compute in Hin. in_cases Hin;
        injection Hin as <- <-; cbn in Hw; injection Hw as <-; left;
        eexists; reflexivity. }
    split.
    { intros a (_ & [Has | Hsrc] & _) _;
        [ vm_compute in Has; discriminate Has | ].
      cbn [s_sources solution_of_tables] in Hsrc.
      apply mem_pair_In in Hsrc. vm_compute in Hsrc. in_cases Hsrc.
      injection Hsrc as <-. reflexivity. }
    apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
    vm_compute. reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hrw | ].
  exists
    (match run 3 (initial_state ReprTypesExample.rebuilt) with
     | Step t => t
     | _ => initial_state ReprTypesExample.rebuilt
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  split.
  - exists [(70%positive, 52%positive)],
      [(80%positive,
        Value_block 0
          [Value_const (Tagged_immediate 7); Value_const (Poison Value)])],
      [(81%positive, Value_const (Tagged_immediate 7))].
    split; [ reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    vm_compute. constructor; [ split; reflexivity | constructor ].
  - apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
    vm_compute. reflexivity.
Qed.

Theorem repr_types_example : repr_types_example_statement.
Proof. apply repr_types_example_gen. vm_compute. reflexivity. Qed.

Theorem repr_types_example_landed : repr_types_example_landed_statement.
Proof. apply repr_types_example_gen. vm_compute. reflexivity. Qed.
