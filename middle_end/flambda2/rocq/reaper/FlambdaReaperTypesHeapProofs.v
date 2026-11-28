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


(* The proofs of FlambdaReaperTypesHeapSpec.v. Nothing here needs review.

   Where the proofs of FlambdaReaperTypesSpec.v need [in_flow_fragment]:
   not in the flow, whose soundness is of [in_heap_fragment] and gives
   [statics_allocated], nor in the meaning of types, which reads no heap,
   but at the exit of the simulation: [rebuild_exit_relation] asks the
   same heap at the two exits, [rebuilt_value] has no case for a location
   (FlambdaReaperTypesProofs.statics_rel_rebuilt asks that the static
   environment hold none), and [closures_exact_of_bound_once] asks
   [in_flow_fragment], of which it reads only that the unit has no older
   form of set of closures.

   The statements of a graph and its tables follow from those without the
   heap by erasing the locations of both runs ([erase_locations]): the
   erasure keeps what types say ([type_satisfied_erase_all]) and makes of
   [rebuilt_value_heap] [rebuilt_value] ([rebuilt_value_heap_erase]). The
   exit relation is that of the simulation with the heap
   (FlambdaReaperTypesHeapExitProofs.v), its [rel] giving
   [rebuilt_value_heap] with [kept_beta] ([rel_rebuilt_value_heap]). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperRebuild FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperLeastProofs
  FlambdaReaperLocalFields FlambdaReaperTypes FlambdaReaperTypesSpec
  FlambdaReaperTypesProofs FlambdaReaperTypesTermSpec
  FlambdaReaperTypesHeapSpec FlambdaReaperTypesUsedValuesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaEnvAliases
  FlambdaEnvAliasesSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs FlambdaClosuresProofs
  FlambdaReaperFlowProofs FlambdaReaperRebuildProofs
  FlambdaReaperClosuresExactProofs FlambdaReaperTypesTermProofs
  FlambdaReaperTypesHeapExitProofs.
Import ListNotations.

Module RBP := FlambdaReaperRebuildProofs.
Module CE := FlambdaReaperClosuresExactProofs.
Module HE := FlambdaReaperTypesHeapExitProofs.

(* ======================================================================== *)
(* The erasure of locations                                                 *)
(* ======================================================================== *)

Local Abbreviation er := erase_locations.

Lemma erase_slots_unfold (slots : list (value_slot * value)) :
  map (fun '(w, x) => (w, er x)) slots = erase_slots slots.
Proof. reflexivity. Qed.

Lemma erase_kind (v : value) : kind_of_value (er v) = kind_of_value v.
Proof. destruct v; reflexivity. Qed.

Lemma erase_not_location (v : value) (l : location) :
  er v <> Value_location l.
Proof. destruct v; discriminate. Qed.

Lemma erase_const (v : value) (c : const) :
  er v = Value_const c <-> v = Value_const c.
Proof.
  destruct v; cbn [erase_locations]; split; intros H;
    try discriminate H; exact H.
Qed.

Lemma map_erase_inj (l : list value) :
  Forall (fun v => ∀ w : value, er v = er w -> v = w) l ->
  ∀ l' : list value, map er l = map er l' -> l = l'.
Proof.
  induction 1 as [ | v l Hv _ IH]; intros [ | v' l'] H;
    cbn [map] in H; try discriminate H; [ reflexivity | ].
  injection H as H1 H2. rewrite (Hv v' H1), (IH l' H2). reflexivity.
Qed.

Lemma erase_slots_inj (slots : list (value_slot * value)) :
  Forall (fun p => ∀ w : value, er (snd p) = er w -> snd p = w) slots ->
  ∀ slots' : list (value_slot * value),
    erase_slots slots = erase_slots slots' -> slots = slots'.
Proof.
  induction 1 as [ | [w x] slots Hx _ IH]; intros [ | [w' x'] slots'] H;
    cbn in H; try discriminate H; [ reflexivity | ].
  injection H as -> H1 H2. cbn [snd] in Hx.
  rewrite (Hx x' H1), (IH slots' H2). reflexivity.
Qed.

(* Two values are one exactly when their erasures are. *)
Lemma erase_inj : ∀ v w : value, er v = er w -> v = w.
Proof.
  intros v. pattern v. apply FlambdaHeapProofs.value_induction; clear v.
  - intros c w H. symmetry. apply erase_const. symmetry. exact H.
  - intros code slots IH [ | code' slots' | | | | | | | ] H; cbn in H;
      try discriminate H.
    injection H as -> H. rewrite (erase_slots_inj slots IH slots' H).
    reflexivity.
  - intros slot decls slots IH [ | | slot' decls' slots' | | | | | | ] H;
      cbn in H; try discriminate H.
    injection H as -> -> H. rewrite (erase_slots_inj slots IH slots' H).
    reflexivity.
  - intros s [ | | | s' | | | | | ] H; cbn in H; try discriminate H.
    exact H.
  - intros tag fields IH [ | | | | tag' fields' | | | | ] H; cbn in H;
      try discriminate H.
    injection H as -> H. rewrite (map_erase_inj fields IH fields' H).
    reflexivity.
  - intros elements IH [ | | | | | elements' | | l' | ] H; cbn in H;
      try discriminate H.
    + injection H as H. rewrite (map_erase_inj elements IH elements' H).
      reflexivity.
    + injection H as H. destruct elements as [ | el [ | ]]; try discriminate H.
      injection H as H. destruct (erase_not_location el l' H).
  - intros c [ | | | | | | c' | | ] H; cbn in H; try discriminate H.
    exact H.
  - intros l [ | | | | | elements' | | l' | ] H; cbn in H;
      try discriminate H.
    + injection H as H. destruct elements' as [ | el [ | ]];
        try discriminate H.
      injection H as H. destruct (erase_not_location el l (eq_sym H)).
    + injection H as ->. reflexivity.
  - intros callee args arity IHc IHa [ | | | | | | | | callee' args' arity']
      H; cbn in H; try discriminate H.
    injection H as H1 H2 ->.
    rewrite (IHc callee' H1), (map_erase_inj args IHa args' H2).
    reflexivity.
Qed.

Lemma option_map_erase (o : option value) (v : value) :
  option_map er o = Some (er v) <-> o = Some v.
Proof.
  destruct o as [w | ]; cbn [option_map]; split; intros H;
    try discriminate H.
  - injection H as H. rewrite (erase_inj w v H). reflexivity.
  - injection H as ->. reflexivity.
Qed.

Lemma option_map_erase_const (o : option value) (c : const) :
  option_map er o = Some (Value_const c) <-> o = Some (Value_const c).
Proof.
  change (Value_const c) with (er (Value_const c)) at 1.
  exact (option_map_erase o (Value_const c)).
Qed.

Lemma lookup_erase_env (vars : env) (x : variable) :
  lookup x (erase_env vars) = option_map er (lookup x vars).
Proof.
  induction vars as [ | [y v] vars IH]; [ reflexivity | ].
  cbn [erase_env map lookup]. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma eval_simple_erase (vars : env) (s : simple) :
  eval_simple (erase_env vars) s = option_map er (eval_simple vars s).
Proof.
  destruct s as [x | sym | c]; cbn [eval_simple];
    [ exact (lookup_erase_env vars x) | reflexivity | reflexivity ].
Qed.

Lemma value_of_name_erase (vars : env) (n : name) :
  value_of_name (erase_env vars) n = option_map er (value_of_name vars n).
Proof. unfold value_of_name. apply eval_simple_erase. Qed.

Lemma variable_is_erase (vars : env) (x : option variable) (n : Z) :
  variable_is (erase_env vars) x n <-> variable_is vars x n.
Proof.
  destruct x as [x | ]; cbn [variable_is]; [ | reflexivity ].
  rewrite lookup_erase_env. apply option_map_erase_const.
Qed.

Lemma lookup_erase_statics (statics : static_env) (s : symbol) :
  lookup s (static_values (erase_statics statics))
  = option_map erase_static (lookup s (static_values statics)).
Proof.
  destruct statics as [code values]. cbn [erase_statics static_values].
  induction values as [ | [s' a] values IH]; [ reflexivity | ].
  cbn [map lookup]. destruct (Pos.eqb s s'); [ reflexivity | exact IH ].
Qed.

Lemma lookup_erase_slots (slots : list (value_slot * value)) (w : value_slot)
    : lookup w (erase_slots slots) = option_map er (lookup w slots).
Proof.
  induction slots as [ | [w' x] slots IH]; [ reflexivity | ].
  cbn [erase_slots map lookup]. destruct (Pos.eqb w w'); [ reflexivity | ].
  exact IH.
Qed.

Lemma domain_erase_slots (slots : list (value_slot * value)) :
  domain (erase_slots slots) = domain slots.
Proof.
  unfold domain. f_equal.
  induction slots as [ | [w x] slots IH]; [ reflexivity | ].
  cbn [erase_slots map fst]. f_equal. exact IH.
Qed.

Section Observations.

Variable statics : static_env.

Local Abbreviation es := (erase_statics statics).

Lemma block_of_erase (v : value) :
  block_of es (er v)
  = option_map (fun p => (fst p, map er (snd p))) (block_of statics v).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args arity];
    try reflexivity.
  cbn [erase_locations block_of]. rewrite lookup_erase_statics.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma block_of_erase_some (v : value) (tag : Z) (fields0 : list value) :
  block_of es (er v) = Some (tag, fields0) <->
  ∃ fields : list value,
    block_of statics v = Some (tag, fields) ∧ fields0 = map er fields.
Proof.
  rewrite block_of_erase. destruct (block_of statics v) as [[tag' fields] | ];
    cbn [option_map fst snd]; split.
  - intros [= -> <-]. exists fields. split; reflexivity.
  - intros (fields' & [= -> ->] & ->). reflexivity.
  - discriminate.
  - intros (fields' & H & _). discriminate H.
Qed.

Lemma boxed_number_of_erase (v : value) :
  boxed_number_of es (er v) = boxed_number_of statics v.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args arity];
    try reflexivity.
  cbn [erase_locations boxed_number_of]. rewrite lookup_erase_statics.
  destruct (lookup s (static_values statics)) as [[ | | | contents | | | ] | ];
    try reflexivity.
  destruct contents; reflexivity.
Qed.

Lemma string_of_erase (v : value) : string_of es (er v) = string_of statics v.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args arity];
    try reflexivity.
  cbn [erase_locations string_of]. rewrite lookup_erase_statics.
  destruct (lookup s (static_values statics)) as [[] | ]; reflexivity.
Qed.

Lemma closure_in_set_erase (v : value) :
  closure_in_set es (er v)
  = option_map (fun p => (fst (fst p), snd (fst p), erase_slots (snd p)))
      (closure_in_set statics v).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args arity];
    try reflexivity.
  - cbn [erase_locations closure_in_set].
    destruct (lookup slot decls); reflexivity.
  - cbn [erase_locations closure_in_set]. rewrite lookup_erase_statics.
    destruct (lookup s (static_values statics)) as [[] | ]; try reflexivity.
    cbn [erase_static option_map]. destruct (lookup slot function_decls);
      reflexivity.
Qed.

Lemma closure_in_set_erase_some (v : value) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (slots0 : list (value_slot * value)) :
  closure_in_set es (er v) = Some (slot, decls, slots0) <->
  ∃ slots : list (value_slot * value),
    closure_in_set statics v = Some (slot, decls, slots)
    ∧ slots0 = erase_slots slots.
Proof.
  rewrite closure_in_set_erase.
  destruct (closure_in_set statics v) as [[[slot' decls'] slots] | ];
    cbn [option_map fst snd]; split.
  - intros [= -> -> <-]. exists slots. split; reflexivity.
  - intros (slots' & [= -> -> ->] & ->). reflexivity.
  - discriminate.
  - intros (slots' & H & _). discriminate H.
Qed.

Lemma project_function_slot_erase (move_from move_to : function_slot)
    (v : value) :
  project_function_slot es move_from move_to (er v)
  = option_map er (project_function_slot statics move_from move_to v).
Proof.
  unfold project_function_slot.
  destruct v as [c | code slots | slot decls slots | s | tag fields
                | elements | c | l | callee args arity];
    cbn [erase_locations]; try reflexivity.
  - destruct (Pos.eqb move_from unnamed_function_slot);
      [ destruct (Pos.eqb move_to unnamed_function_slot) | ]; reflexivity.
  - destruct (Pos.eqb slot move_from); [ | reflexivity ].
    destruct (lookup slot decls); [ | reflexivity ].
    destruct (lookup move_to decls); reflexivity.
  - rewrite lookup_erase_statics.
    destruct (lookup s (static_values statics)) as [[] | ]; cbn [option_map];
      try reflexivity.
    + destruct (Pos.eqb move_from unnamed_function_slot);
        [ destruct (Pos.eqb move_to unnamed_function_slot) | ]; reflexivity.
    + cbn [erase_static]. destruct (Pos.eqb slot move_from); [ | reflexivity ].
      destruct (lookup slot function_decls); [ | reflexivity ].
      destruct (lookup move_to function_decls); [ | reflexivity ].
      destruct (lookup move_to closure_symbols); reflexivity.
Qed.

Lemma relation_holds_erase (relation : Relation.t) (w : value) (n : Z) :
  relation_holds es relation (er w) n <-> relation_holds statics relation w n.
Proof.
  destruct relation; cbn [relation_holds]; unfold is_immediate, is_block.
  - rewrite erase_kind, erase_const; reflexivity.
  - split; intros [[(i & Hi) Hn] | [(tag & fields & Hb) Hn]].
    + left. split; [ exists i; apply erase_const; exact Hi | exact Hn ].
    + right. split; [ | exact Hn ].
      destruct (proj1 (block_of_erase_some w tag fields) Hb)
        as (fields' & Hb' & _).
      exists tag, fields'. exact Hb'.
    + left. split; [ exists i; apply erase_const; exact Hi | exact Hn ].
    + right. split; [ | exact Hn ]. exists tag, (map er fields).
      apply block_of_erase_some. exists fields. split; [ exact Hb | ].
      reflexivity.
  - split; intros [(i & Hi) | [(fields & Hb) Ht]].
    + left. exists i. apply erase_const. exact Hi.
    + right. split; [ | exact Ht ].
      destruct (proj1 (block_of_erase_some w n fields) Hb)
        as (fields' & Hb' & _).
      exists fields'. exact Hb'.
    + left. exists i. apply erase_const. exact Hi.
    + right. split; [ | exact Ht ]. exists (map er fields).
      apply block_of_erase_some. exists fields. split; [ exact Hb | ].
      reflexivity.
Qed.

End Observations.

(* ======================================================================== *)
(* The meaning of types does not see the erasure                            *)
(* ======================================================================== *)

Section Erase_meaning.

Variable statics : static_env.
Variable vars : env.

Local Abbreviation es := (erase_statics statics).
Local Abbreviation ev := (erase_env vars).

Lemma descr_erase {head : Type} (k : flambda_kind) (hs0 hs : head -> Prop)
    (v : value) (ty : type_descr head) :
  (∀ h : head, ty = Ok (No_alias h) -> (hs0 h <-> hs h)) ->
  descr_satisfied ev k hs0 (er v) ty <-> descr_satisfied vars k hs v ty.
Proof.
  intros H. destruct ty as [ | [h | s] | ]; cbn [descr_satisfied].
  - rewrite erase_kind. reflexivity.
  - exact (H h eq_refl).
  - rewrite eval_simple_erase, option_map_erase, erase_kind. reflexivity.
  - reflexivity.
Qed.

Lemma fields_satisfied_erase (tys : list flambda_type) :
  (∀ ty : flambda_type, ty ∈ tys -> ∀ v : value,
     type_satisfied es ev (er v) ty <-> type_satisfied statics vars v ty) ->
  ∀ fields : list value,
    fields_satisfied (type_satisfied es ev) tys (map er fields)
    <-> fields_satisfied (type_satisfied statics vars) tys fields.
Proof.
  induction tys as [ | ty tys IH]; intros H fields; [ reflexivity | ].
  destruct fields as [ | f fields]; cbn [map fields_satisfied];
    [ reflexivity | ].
  rewrite (H ty (or_introl eq_refl) f),
    (IH (fun ty' Hin => H ty' (or_intror Hin)) fields).
  reflexivity.
Qed.

Lemma block_index_erase (tag : Z) (fields : list value)
    (idx : row_like_index nat block_shape) :
  block_index_satisfied tag (map er fields) idx
  <-> block_index_satisfied tag fields idx.
Proof.
  unfold block_index_satisfied, block_has_shape. rewrite length_map, map_map.
  rewrite (map_ext (fun x => kind_of_value (er x)) kind_of_value
             erase_kind fields).
  reflexivity.
Qed.

Lemma closures_index_erase (functions : list (function_slot * code_id))
    (slots : list (value_slot * value))
    (idx : row_like_index set_of_closures_contents unit) :
  closures_index_satisfied functions (erase_slots slots) idx
  <-> closures_index_satisfied functions slots idx.
Proof.
  unfold closures_index_satisfied. rewrite domain_erase_slots. reflexivity.
Qed.

Lemma type_satisfied_erase_all :
  (∀ (t : flambda_type) (v : value),
     type_satisfied es ev (er v) t <-> type_satisfied statics vars v t)
  ∧ (∀ (head : head_of_kind_value) (v : value),
       head_of_kind_value_satisfied es ev (er v) head
       <-> head_of_kind_value_satisfied statics vars v head)
  ∧ (∀ (head : head_of_kind_value_non_null) (v : value),
       head_of_kind_value_non_null_satisfied es ev (er v) head
       <-> head_of_kind_value_non_null_satisfied statics vars v head)
  ∧ (∀ (blocks : row_like_for_blocks) (tag : Z) (fields : list value),
       row_like_for_blocks_satisfied es ev tag (map er fields) blocks
       <-> row_like_for_blocks_satisfied statics vars tag fields blocks)
  ∧ (∀ (closures : row_like_for_closures) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (slots : list (value_slot * value)),
       row_like_for_closures_satisfied es ev (er v) slot functions
         (erase_slots slots) closures
       <-> row_like_for_closures_satisfied statics vars v slot functions slots
             closures)
  ∧ (∀ (entry : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (slots : list (value_slot * value)),
       closures_entry_satisfied es ev (er v) slot functions (erase_slots slots)
         entry
       <-> closures_entry_satisfied statics vars v slot functions slots entry)
  ∧ (∀ extension : env_extension,
       env_extension_satisfied es ev extension
       <-> env_extension_satisfied statics vars extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead v. rewrite !type_satisfied_value.
    apply descr_erase. intros h E. subst ty. exact (Hhead v).
  - (* Type_naked_immediate *)
    intros ty v. rewrite !type_satisfied_naked_immediate.
    apply descr_erase. intros h _.
    unfold head_of_kind_naked_immediate_satisfied.
    split; intros (i & Hv & Hnum & Hrel); exists i;
      (split; [ | split; [ exact Hnum | ] ]).
    + exact (proj1 (erase_const v _) Hv).
    + intros relation n Hin. destruct (Hrel relation n Hin) as (w & Hw & Hh).
      rewrite value_of_name_erase in Hw.
      destruct (value_of_name vars n) as [w0 | ] eqn:E; cbn in Hw;
        [ | discriminate Hw ].
      injection Hw as <-. exists w0.
      split; [ reflexivity | exact (proj1 (relation_holds_erase _ _ _ _) Hh) ].
    + exact (proj2 (erase_const v _) Hv).
    + intros relation n Hin. destruct (Hrel relation n Hin) as (w & Hw & Hh).
      exists (er w). rewrite value_of_name_erase, Hw.
      split; [ reflexivity | exact (proj2 (relation_holds_erase _ _ _ _) Hh) ].
  - (* Type_naked_number *)
    intros k ty v. rewrite !type_satisfied_naked_number.
    apply descr_erase. intros numbers _. unfold naked_numbers_satisfied.
    split; intros (c & n & Hv & Hk & Hn & Hin); exists c, n;
      (split; [ | auto ]).
    + exact (proj1 (erase_const v _) Hv).
    + exact (proj2 (erase_const v _) Hv).
  - intros ty v. rewrite !type_satisfied_rec_info.
    apply descr_erase. intros _ _. rewrite erase_kind. reflexivity.
  - intros ty v. rewrite !type_satisfied_region.
    apply descr_erase. intros _ _. rewrite erase_kind. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnn v. rewrite !head_of_kind_value_satisfied_eq.
    destruct non_null as [ | h | ]; [ | rewrite (Hnn v) | ];
      destruct is_null as [ | x]; rewrite ?variable_is_erase, ?erase_kind,
        ?erase_const; reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himm
      Hblocks Hext v.
    rewrite !variant_satisfied.
    split; intros [(i & Hv & Hi & Him & He)
                  | (tag & fields & Hb & Hi & Ht & Hbl & He)].
    + left. exists i.
      split; [ exact (proj1 (erase_const v _) Hv) | ].
      split; [ exact (proj1 (variable_is_erase vars is_int 1) Hi) | ].
      split.
      * destruct immediates as [ty | ]; [ | exact I ].
        exact (proj1 (Himm (Value_const (Naked_immediate i))) Him).
      * destruct extensions as [ | wi wb]; [ exact I | ].
        exact (proj1 (proj1 Hext) He).
    + right. destruct (proj1 (block_of_erase_some statics v tag fields) Hb)
        as (fields0 & Hb0 & ->).
      exists tag, fields0.
      split; [ exact Hb0 | ].
      split; [ exact (proj1 (variable_is_erase vars is_int 0) Hi) | ].
      split; [ exact (proj1 (variable_is_erase vars get_tag tag) Ht) | ].
      split.
      * destruct blocks as [bl | ]; [ | exact I ].
        exact (proj1 (Hblocks tag fields0) Hbl).
      * destruct extensions as [ | wi wb]; [ exact I | ].
        exact (proj1 (proj2 Hext) He).
    + left. exists i.
      split; [ exact (proj2 (erase_const v _) Hv) | ].
      split; [ exact (proj2 (variable_is_erase vars is_int 1) Hi) | ].
      split.
      * destruct immediates as [ty | ]; [ | exact I ].
        exact (proj2 (Himm (Value_const (Naked_immediate i))) Him).
      * destruct extensions as [ | wi wb]; [ exact I | ].
        exact (proj2 (proj1 Hext) He).
    + right. exists tag, (map er fields).
      split.
      { apply block_of_erase_some. exists fields.
        split; [ exact Hb | reflexivity ]. }
      split; [ exact (proj2 (variable_is_erase vars is_int 0) Hi) | ].
      split; [ exact (proj2 (variable_is_erase vars get_tag tag) Ht) | ].
      split.
      * destruct blocks as [bl | ]; [ | exact I ].
        exact (proj2 (Hblocks tag fields) Hbl).
      * destruct extensions as [ | wi wb]; [ exact I | ].
        exact (proj2 (proj2 Hext) He).
  - (* Mutable_block *)
    intros am v. rewrite !mutable_block_head_satisfied. reflexivity.
  - (* Boxed_number *)
    intros k contents am Hc v. rewrite !boxed_number_satisfied,
      boxed_number_of_erase.
    split; intros (c & Hb & Hk & Hs); exists c;
      (split; [ exact Hb | split; [ exact Hk | ] ]).
    + exact (proj1 (Hc (Value_const c)) Hs).
    + exact (proj2 (Hc (Value_const c)) Hs).
  - (* Closures *)
    intros bfs am Hcl v. rewrite !closures_satisfied.
    split; intros (slot & functions & slots & Hcis & Hrow).
    + destruct (proj1 (closure_in_set_erase_some statics v slot functions
                         slots) Hcis)
        as (slots0 & Hcis0 & ->).
      exists slot, functions, slots0.
      split; [ exact Hcis0 | exact (proj1 (Hcl v slot functions slots0) Hrow) ].
    + exists slot, functions, (erase_slots slots).
      split; [ | exact (proj2 (Hcl v slot functions slots) Hrow) ].
      apply closure_in_set_erase_some. exists slots.
      split; [ exact Hcis | reflexivity ].
  - (* String *)
    intros strs v. rewrite !string_satisfied, string_of_erase. reflexivity.
  - (* Row_like_for_blocks *)
    intros known other am Hknown Hother tag fields.
    rewrite !row_like_for_blocks_find_iff.
    assert (Hcase :
      ∀ case : row_like_block_case,
        (∀ ty : flambda_type, ty ∈ case_maps_to case -> ∀ v : value,
           type_satisfied es ev (er v) ty
           <-> type_satisfied statics vars v ty) ->
        (env_extension_satisfied es ev (case_env_extension case)
         <-> env_extension_satisfied statics vars (case_env_extension case)) ->
        (row_like_block_case_satisfied es ev tag (map er fields) case
         <-> row_like_block_case_satisfied statics vars tag fields case)).
    { intros case Htys Hx.
      unfold row_like_block_case_satisfied, block_case_satisfied.
      rewrite block_index_erase, (fields_satisfied_erase _ Htys fields), Hx.
      reflexivity. }
    destruct (Keyed.lookup Z.eqb tag known) as [[case | ] | ] eqn:Hfind.
    + destruct (Hknown tag case (keyed_lookup_in Z.eqb Z.eqb_eq _ _ _ Hfind))
        as [Htys Hx].
      exact (Hcase case Htys Hx).
    + reflexivity.
    + destruct other as [case | ]; [ | reflexivity ].
      exact (Hcase case (proj1 Hother) (proj2 Hother)).
  - (* Row_like_for_closures *)
    intros known Hknown v slot functions slots.
    rewrite !row_like_for_closures_lookup_iff.
    assert (Hiff :
      ∀ case : row_like_closures_case,
        lookup slot known = Some case ->
        (row_like_closures_case_satisfied es ev (er v) slot functions
           (erase_slots slots) case
         <-> row_like_closures_case_satisfied statics vars v slot functions
               slots case)).
    { intros case Hl.
      destruct (Hknown slot case (lookup_in _ _ _ Hl)) as [He Hx].
      unfold row_like_closures_case_satisfied, closures_case_satisfied.
      rewrite closures_index_erase, (He v slot functions slots), Hx.
      reflexivity. }
    split; intros (case & Hl & Hc); exists case;
      (split; [ exact Hl | ]).
    + exact (proj1 (Hiff case Hl) Hc).
    + exact (proj2 (Hiff case Hl) Hc).
  - (* Closures_entry *)
    intros fts cts vts Hct Hvt v slot functions slots.
    rewrite !closures_entry_lookup_iff.
    split; intros [Hf [Hc Hs]]; (split; [ exact Hf | split ]).
    + intros g ty closure Hl Hp.
      apply (proj1 (Hct g ty (lookup_in _ _ _ Hl) closure)).
      apply (Hc g ty (er closure) Hl).
      rewrite project_function_slot_erase, Hp. reflexivity.
    + intros w ty contents Hl Hlc.
      apply (proj1 (Hvt w ty (lookup_in _ _ _ Hl) contents)).
      apply (Hs w ty (er contents) Hl).
      rewrite lookup_erase_slots, Hlc. reflexivity.
    + intros g ty closure0 Hl Hp. rewrite project_function_slot_erase in Hp.
      destruct (project_function_slot statics slot g v) as [closure | ] eqn:E;
        cbn [option_map] in Hp; [ | discriminate Hp ].
      injection Hp as <-.
      apply (proj2 (Hct g ty (lookup_in _ _ _ Hl) closure)).
      exact (Hc g ty closure Hl E).
    + intros w ty c0 Hl Hlc. rewrite lookup_erase_slots in Hlc.
      destruct (lookup w slots) as [c | ] eqn:E; cbn [option_map] in Hlc;
        [ | discriminate Hlc ].
      injection Hlc as <-.
      apply (proj2 (Hvt w ty (lookup_in _ _ _ Hl) c)).
      exact (Hs w ty c Hl E).
  - (* Env_extension *)
    intros eqs Htys. rewrite !env_extension_lookup_iff.
    split; intros H n ty Hl; destruct (H n ty Hl) as (w & Hw & Hs);
      pose proof (Htys n ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hl))
        as Hty.
    + rewrite value_of_name_erase in Hw.
      destruct (value_of_name vars n) as [w0 | ] eqn:E; cbn [option_map] in Hw;
        [ | discriminate Hw ].
      injection Hw as <-. exists w0.
      split; [ reflexivity | exact (proj1 (Hty w0) Hs) ].
    + exists (er w). rewrite value_of_name_erase, Hw.
      split; [ reflexivity | exact (proj2 (Hty w) Hs) ].
Qed.

End Erase_meaning.

Theorem type_satisfied_erase : type_satisfied_erase_statement.
Proof.
  intros statics vars v t.
  exact (proj1 (type_satisfied_erase_all statics vars) t v).
Qed.

(* ======================================================================== *)
(* What the statements ask, erased                                          *)
(* ======================================================================== *)

Lemma erase_unallocated (w : value) : unallocated w -> er w = w.
Proof. intros [(c & ->) | (s & ->)]; reflexivity. Qed.

Lemma type_in_fragment_erase (loose : bool) (vars : env) (t : flambda_type) :
  type_in_fragment_with loose vars t ->
  type_in_fragment_with loose (erase_env vars) t.
Proof.
  intros [Hs Hal]. split; [ exact Hs | ].
  intros s k w0 Hin Hw0. rewrite eval_simple_erase in Hw0.
  destruct (eval_simple vars s) as [w | ] eqn:E; cbn [option_map] in Hw0;
    [ | discriminate Hw0 ].
  injection Hw0 as <-. pose proof (Hal s k w Hin E) as Hu.
  rewrite (erase_unallocated w Hu). exact Hu.
Qed.

Lemma env_in_fragment_erase (loose : bool) (vars : env)
    (tenv : Typing_env.t) :
  env_in_fragment_with loose vars tenv ->
  env_in_fragment_with loose (erase_env vars) tenv.
Proof.
  intros H n ty bt Hl.
  exact (type_in_fragment_erase loose vars ty (H n ty bt Hl)).
Qed.

Lemma typing_env_satisfied_strictly_erase (statics : static_env)
    (vars : env) (tenv : Typing_env.t) :
  typing_env_satisfied_strictly statics vars tenv ->
  typing_env_satisfied_strictly (erase_statics statics) (erase_env vars) tenv.
Proof.
  intros (Hb & Htypes & Hcan & Hlisted).
  split; [ exact Hb | split ].
  - intros n ty bt v0 Hl Hv0. rewrite value_of_name_erase in Hv0.
    destruct (value_of_name vars n) as [v | ] eqn:E; cbn [option_map] in Hv0;
      [ | discriminate Hv0 ].
    injection Hv0 as <-.
    destruct (Htypes n ty bt v Hl E) as [[[] _] | Hs].
    right. exact (proj2 (type_satisfied_erase statics vars v ty) Hs).
  - split.
    + intros n c Hd. rewrite value_of_name_erase, eval_simple_erase,
        (Hcan n c Hd).
      reflexivity.
    + intros c n Hl. rewrite value_of_name_erase, eval_simple_erase,
        (Hlisted c n Hl).
      reflexivity.
Qed.

Lemma copies_valued_erase (tenv : Typing_env.t) (canon : canon_fn)
    (vars vars' : env) :
  copies_valued tenv canon vars vars' ->
  copies_valued tenv canon (erase_env vars) (erase_env vars').
Proof.
  intros H s x k m Hc.
  rewrite value_of_name_erase, eval_simple_erase, (H s x k m Hc).
  reflexivity.
Qed.

Lemma statics_allocated_erase (sol : solution) (statics : static_env) :
  statics_allocated sol statics ->
  statics_allocated sol (erase_statics statics).
Proof.
  intros H s a0 Hl. rewrite lookup_erase_statics in Hl.
  destruct (lookup s (static_values statics)) as [a | ] eqn:E;
    cbn [option_map] in Hl; [ exact (H s a E) | discriminate Hl ].
Qed.

Lemma static_sets_whole_erase (sol : solution)
    (statics statics' : static_env) :
  static_sets_whole sol statics statics' ->
  static_sets_whole sol (erase_statics statics) (erase_statics statics').
Proof.
  intros H s slot syms decls vs0 Hl Hhu. rewrite lookup_erase_statics in Hl.
  destruct (lookup s (static_values statics)) as [a | ] eqn:E;
    cbn [option_map] in Hl; [ | discriminate Hl ].
  destruct a; cbn [erase_static] in Hl; try discriminate Hl.
  injection Hl as <- <- <- <-.
  destruct (H s _ _ _ _ E Hhu) as [Hslot (vs' & Hall)].
  split; [ exact Hslot | ]. exists (erase_slots vs').
  intros g s_g Hg. destruct (Hall g s_g Hg) as [H1 H2].
  rewrite !lookup_erase_statics, H1, H2. split; reflexivity.
Qed.

Lemma fields_rebuilt_map (P Q : nat -> value -> value -> Prop)
    (f : value -> value) (fields : list value) :
  Forall (fun x => ∀ (i : nat) (x' : value), P i x x' -> Q i (f x) (f x'))
    fields ->
  ∀ (k : nat) (fields' : list value),
    fields_rebuilt P k fields fields' ->
    fields_rebuilt Q k (map f fields) (map f fields').
Proof.
  induction 1 as [ | x l Hx _ IH]; intros k [ | x' l'] H;
    cbn [fields_rebuilt map] in H |- *; try contradiction; [ exact I | ].
  destruct H as [H1 H2]. split; [ exact (Hx k x' H1) | exact (IH (S k) l' H2) ].
Qed.

Section Rebuilt_erase.

Variable G : graph.
Variable sol : solution.
Variable β : location -> location -> Prop.

Lemma slots_rebuilt_erase (decls : list (function_slot * code_id))
    (sites : function_slot -> node) (vs vs' : list (value_slot * value))
    (P : value -> node -> value -> Prop) :
  Forall (fun p => ∀ (n : node) (x' : value),
            P (snd p) n x' -> rebuilt_value G sol (er (snd p)) n (er x')) vs ->
  for_all_bindings (value_slot_rebuilt G sol P decls sites vs') vs ->
  for_all_bindings
    (value_slot_rebuilt G sol (rebuilt_value G sol) decls sites
       (erase_slots vs'))
    (erase_slots vs).
Proof.
  intros IH H. apply (proj2 (for_all_bindings_lookup _ _ _)).
  intros w x0 Hl. rewrite lookup_erase_slots in Hl.
  destruct (lookup w vs) as [x | ] eqn:E; cbn [option_map] in Hl;
    [ | discriminate Hl ].
  injection Hl as <-.
  pose proof (proj1 (for_all_bindings_lookup _ _ _) H w x E) as Hx.
  intros f id Hf Hused. destruct (Hx f id Hf Hused)
    as (from & x' & Hheld & Hhs & Hhu & Hl' & Hrv).
  exists from, (er x').
  split; [ exact Hheld | split; [ exact Hhs | split; [ exact Hhu | ] ] ].
  split; [ rewrite lookup_erase_slots, Hl'; reflexivity | ].
  rewrite Forall_forall in IH.
  exact (IH (w, x) (lookup_in _ _ _ E) from x' Hrv).
Qed.

(* A value rebuilt with the heap, erased, is rebuilt. *)
Lemma rebuilt_value_heap_erase :
  ∀ (v : value) (n : node) (v' : value),
    rebuilt_value_heap G sol β v n v' ->
    rebuilt_value G sol (er v) n (er v').
Proof.
  intros v. pattern v. apply FlambdaHeapProofs.value_induction; clear v;
    cbn [rebuilt_value_heap rebuilt_value erase_locations].
  - intros c n v' [Has ->]. split; [ exact Has | reflexivity ].
  - intros code slots _ n v' [].
  - intros slot decls slots IH n v' (sites & vs' & -> & Hmay & Hsites & Hall).
    exists sites, (erase_slots vs').
    split; [ reflexivity | split; [ exact Hmay | split; [ exact Hsites | ] ] ].
    exact (slots_rebuilt_erase decls sites slots vs'
             (rebuilt_value_heap G sol β) IH Hall).
  - intros s n v' (-> & Hmay & Hhu). split; [ reflexivity | auto ].
  - intros tag fields IH n v' (a & fields' & Hmay & Hhs & -> & Hf).
    exists a, (map er fields').
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    refine (fields_rebuilt_map _ _ er fields _ 0 fields' Hf).
    revert IH. apply Forall_impl. intros x IHx i x' (from & Hc & Hu & Hn).
    rewrite erase_kind. exists from. split; [ exact Hc | split ].
    + intros Hused. destruct (Hu Hused) as [Hhu Hrv].
      exact (conj Hhu (IHx from x' Hrv)).
    + intros Hunused. rewrite (Hn Hunused). reflexivity.
  - intros elements _ n v' (elements' & ->).
    exists (map er elements'). reflexivity.
  - intros c n v' (a & bn & from & Hmay & Hhs & Hk & Hc & ->).
    exists a, bn, from. repeat (split; [ assumption | ]).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros l n v' (l' & -> & _). exists [Value_location l']. reflexivity.
  - intros callee args arity _ _ n v' (closure' & args' & ->).
    exists (er closure'), (map er args'). reflexivity.
Qed.

Lemma rebuilt_under_heap_erase (m : rewrite_metadata) (v v' : value) :
  rebuilt_under_heap G sol β m v v' -> rebuilt_under G sol m (er v) (er v').
Proof.
  intros [(c & -> & ->) | (n & Hv & Hat & Hhu)].
  - left. exists c. split; reflexivity.
  - right. exists n.
    exact (conj (rebuilt_value_heap_erase v n v' Hv) (conj Hat Hhu)).
Qed.

Lemma statics_rebuilt_heap_erase (statics statics' : static_env) :
  statics_rebuilt_heap G sol β statics statics' ->
  statics_rebuilt G sol (erase_statics statics) (erase_statics statics').
Proof.
  intros H s a0 Hl Hhu. rewrite lookup_erase_statics in Hl.
  destruct (lookup s (static_values statics)) as [a | ] eqn:E;
    cbn [option_map] in Hl; [ | discriminate Hl ].
  injection Hl as <-. pose proof (H s a E Hhu) as Hs.
  rewrite lookup_erase_statics.
  destruct a as [code vs | slot syms decls vs | tag fields | contents | bytes
                | elements | c];
    cbn [static_rebuilt_heap] in Hs; cbn [erase_static static_rebuilt].
  - exact Hs.
  - destruct Hs as (sites & vs' & -> & Hsite & Hsites & Hall).
    exists sites, (erase_slots vs').
    split; [ reflexivity | split; [ exact Hsite | split; [ exact Hsites | ] ] ].
    apply (slots_rebuilt_erase decls sites vs vs'
             (rebuilt_value_heap G sol β)); [ | exact Hall ].
    apply Forall_forall. intros p _ n x' Hrv.
    exact (rebuilt_value_heap_erase (snd p) n x' Hrv).
  - destruct Hs as (fields' & -> & Hf). exists (map er fields').
    split; [ reflexivity | ].
    refine (fields_rebuilt_map _ _ er fields _ 0 fields' Hf).
    apply Forall_forall. intros x _ i x' (from & Hc & Hu & Hn).
    unfold field_rebuilt. rewrite erase_kind. exists from.
    split; [ exact Hc | split ].
    + intros Hused. destruct (Hu Hused) as [Hhu' Hrv].
      exact (conj Hhu' (rebuilt_value_heap_erase x from x' Hrv)).
    + intros Hunused. rewrite (Hn Hunused). reflexivity.
  - rewrite Hs. reflexivity.
  - rewrite Hs. reflexivity.
  - destruct Hs as (elements' & ->). exists (map er elements'). reflexivity.
  - rewrite Hs. reflexivity.
Qed.

End Rebuilt_erase.

(* ======================================================================== *)
(* The statements of a graph and its tables                                 *)
(* ======================================================================== *)

Lemma rewrite_type_sound_heap_gen (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment_with loose valuation t ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Ht Hv Hsat.
  apply (proj1 (type_satisfied_erase statics' valuation' v' _)).
  apply (rewrite_type_sound_in fx loose Hfx is_local F T sets tenv canon
           (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') fuel m t (er v)
           (er v')).
  - exact Hleast.
  - exact Hlocal.
  - exact Hsets.
  - exact (typing_env_satisfied_strictly_erase _ _ _ Henv).
  - exact (env_in_fragment_erase _ _ _ Hfrag).
  - exact (statics_allocated_erase _ _ Halloc).
  - exact (statics_rebuilt_heap_erase _ _ _ _ _ Hst).
  - intros Hl. exact (static_sets_whole_erase _ _ _ (Hwhole Hl)).
  - exact (copies_valued_erase _ _ _ _ Hcopies).
  - exact (type_in_fragment_erase _ _ _ Ht).
  - exact (rebuilt_under_heap_erase _ _ _ _ _ _ Hv).
  - exact (proj2 (type_satisfied_erase _ _ _ _) Hsat).
Qed.

Lemma rewrite_symbol_type_sound_heap_gen (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (s : symbol) (allocated : statically_allocated_value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Hl.
  apply (proj1 (type_satisfied_erase statics' valuation' (Value_symbol s) _)).
  apply (rewrite_symbol_type_sound_in fx loose Hfx is_local F T sets tenv
           canon (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') s
           (erase_static allocated)).
  - exact Hleast.
  - exact Hlocal.
  - exact Hsets.
  - exact (typing_env_satisfied_strictly_erase _ _ _ Henv).
  - exact (env_in_fragment_erase _ _ _ Hfrag).
  - exact (statics_allocated_erase _ _ Halloc).
  - exact (statics_rebuilt_heap_erase _ _ _ _ _ Hst).
  - intros Hl'. exact (static_sets_whole_erase _ _ _ (Hwhole Hl')).
  - exact (copies_valued_erase _ _ _ _ Hcopies).
  - rewrite lookup_erase_statics, Hl. reflexivity.
Qed.

Lemma rewrite_result_type_sound_heap_gen (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (code_var type_var : variable) (v v' : value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc
    Hst Hwhole Hcopies Hfind Hl Hv.
  apply (proj1 (type_satisfied_erase statics' valuation' v' _)).
  apply (rewrite_result_type_sound_in fx loose Hfx is_local F T sets tenv
           canon (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') code_var type_var
           (er v) (er v')).
  - exact Hleast.
  - exact Hlocal.
  - exact Hsets.
  - exact (typing_env_satisfied_strictly_erase _ _ _ Henv).
  - exact (env_in_fragment_erase _ _ _ Hfrag).
  - exact (statics_allocated_erase _ _ Halloc).
  - exact (statics_rebuilt_heap_erase _ _ _ _ _ Hst).
  - intros Hl'. exact (static_sets_whole_erase _ _ _ (Hwhole Hl')).
  - exact (copies_valued_erase _ _ _ _ Hcopies).
  - exact Hfind.
  - rewrite lookup_erase_env, Hl. reflexivity.
  - exact (rebuilt_value_heap_erase _ _ _ _ _ _ Hv).
Qed.

(* The statements, for the fixes [fx]. *)
Lemma rewrite_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hcopies Ht Hv Hsat.
  exact (rewrite_type_sound_heap_gen fx false Hfx is_local F T sets tenv canon
           statics statics' valuation valuation' β fuel m t v v'
           (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun H => match diff_false_true H with end) Hcopies Ht Hv Hsat).
Qed.

Lemma rewrite_unaliased_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Ht Hv Hsat.
  exact (rewrite_type_sound_heap_gen fx true Hfx is_local F T sets tenv canon
           statics statics' valuation valuation' β fuel m t v v'
           (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun _ => Hwhole) Hcopies Ht Hv Hsat).
Qed.

Lemma rewrite_symbol_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_symbol_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hcopies Hl.
  exact (rewrite_symbol_type_sound_heap_gen fx false Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β s allocated
           (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun H => match diff_false_true H with end) Hcopies Hl).
Qed.

Lemma rewrite_unaliased_symbol_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_symbol_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Hl.
  exact (rewrite_symbol_type_sound_heap_gen fx true Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β s allocated
           (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun _ => Hwhole) Hcopies Hl).
Qed.

Lemma rewrite_result_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_result_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hcopies Hfind Hl Hv.
  exact (rewrite_result_type_sound_heap_gen fx false Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β code_var
           type_var v v' (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc
           Hst (fun H => match diff_false_true H with end) Hcopies Hfind Hl
           Hv).
Qed.

Lemma rewrite_unaliased_result_type_sound_heap_with_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_result_type_sound_heap_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hwhole Hcopies Hfind Hl Hv.
  exact (rewrite_result_type_sound_heap_gen fx true Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β code_var
           type_var v v' (or_introl Hleast) Hlocal Hsets Henv Hfrag Halloc
           Hst (fun _ => Hwhole) Hcopies Hfind Hl Hv).
Qed.

Theorem rewrite_unaliased_result_type_sound_heap :
  rewrite_unaliased_result_type_sound_heap_statement.
Proof.
  exact (rewrite_unaliased_result_type_sound_heap_with_gen _ all_fixes_sound).
Qed.

Theorem rewrite_unaliased_result_type_sound_heap_landed :
  rewrite_unaliased_result_type_sound_heap_landed_statement.
Proof.
  exact (rewrite_unaliased_result_type_sound_heap_with_gen _
           landed_fixes_sound).
Qed.

Theorem rewrite_type_sound_heap : rewrite_type_sound_heap_statement.
Proof. exact (rewrite_type_sound_heap_with_gen _ all_fixes_sound). Qed.

Theorem rewrite_type_sound_heap_landed :
  rewrite_type_sound_heap_landed_statement.
Proof. exact (rewrite_type_sound_heap_with_gen _ landed_fixes_sound). Qed.

Theorem rewrite_symbol_type_sound_heap :
  rewrite_symbol_type_sound_heap_statement.
Proof. exact (rewrite_symbol_type_sound_heap_with_gen _ all_fixes_sound). Qed.

Theorem rewrite_symbol_type_sound_heap_landed :
  rewrite_symbol_type_sound_heap_landed_statement.
Proof.
  exact (rewrite_symbol_type_sound_heap_with_gen _ landed_fixes_sound).
Qed.

Theorem rewrite_result_type_sound_heap :
  rewrite_result_type_sound_heap_statement.
Proof. exact (rewrite_result_type_sound_heap_with_gen _ all_fixes_sound). Qed.

Theorem rewrite_result_type_sound_heap_landed :
  rewrite_result_type_sound_heap_landed_statement.
Proof.
  exact (rewrite_result_type_sound_heap_with_gen _ landed_fixes_sound).
Qed.

Theorem rewrite_unaliased_type_sound_heap :
  rewrite_unaliased_type_sound_heap_statement.
Proof.
  exact (rewrite_unaliased_type_sound_heap_with_gen _ all_fixes_sound).
Qed.

Theorem rewrite_unaliased_type_sound_heap_landed :
  rewrite_unaliased_type_sound_heap_landed_statement.
Proof.
  exact (rewrite_unaliased_type_sound_heap_with_gen _ landed_fixes_sound).
Qed.

Theorem rewrite_unaliased_symbol_type_sound_heap :
  rewrite_unaliased_symbol_type_sound_heap_statement.
Proof.
  exact (rewrite_unaliased_symbol_type_sound_heap_with_gen _ all_fixes_sound).
Qed.

Theorem rewrite_unaliased_symbol_type_sound_heap_landed :
  rewrite_unaliased_symbol_type_sound_heap_landed_statement.
Proof.
  exact (rewrite_unaliased_symbol_type_sound_heap_with_gen _
           landed_fixes_sound).
Qed.

(* With local fields. *)
Lemma rewrite_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hcopies Ht Hv Hsat.
  exact (rewrite_type_sound_heap_gen fx false Hfx is_local F T sets tenv canon
           statics statics' valuation valuation' β fuel m t v v'
           (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun H => match diff_false_true H with end) Hcopies Ht Hv Hsat).
Qed.

Lemma rewrite_unaliased_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Ht Hv Hsat.
  exact (rewrite_type_sound_heap_gen fx true Hfx is_local F T sets tenv canon
           statics statics' valuation valuation' β fuel m t v v'
           (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun _ => Hwhole) Hcopies Ht Hv Hsat).
Qed.

Lemma rewrite_symbol_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_symbol_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hcopies Hl.
  exact (rewrite_symbol_type_sound_heap_gen fx false Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β s allocated
           (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun H => match diff_false_true H with end) Hcopies Hl).
Qed.

Lemma rewrite_unaliased_symbol_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_symbol_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Hl.
  exact (rewrite_symbol_type_sound_heap_gen fx true Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β s allocated
           (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc Hst
           (fun _ => Hwhole) Hcopies Hl).
Qed.

Lemma rewrite_result_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_result_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hcopies Hfind Hl Hv.
  exact (rewrite_result_type_sound_heap_gen fx false Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β code_var
           type_var v v' (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc
           Hst (fun H => match diff_false_true H with end) Hcopies Hfind Hl
           Hv).
Qed.

Lemma rewrite_unaliased_result_type_sound_heap_local_gen (fx : fixes) :
  fixes_sound fx -> rewrite_unaliased_result_type_sound_heap_local_with fx.
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hwhole Hcopies Hfind Hl Hv.
  exact (rewrite_result_type_sound_heap_gen fx true Hfx is_local F T sets
           tenv canon statics statics' valuation valuation' β code_var
           type_var v v' (or_intror Hleast) Hlocal Hsets Henv Hfrag Halloc
           Hst (fun _ => Hwhole) Hcopies Hfind Hl Hv).
Qed.

Theorem rewrite_type_sound_heap_local :
  rewrite_type_sound_heap_local_statement.
Proof. exact (rewrite_type_sound_heap_local_gen _ all_fixes_sound). Qed.

Theorem rewrite_type_sound_heap_local_landed :
  rewrite_type_sound_heap_local_landed_statement.
Proof. exact (rewrite_type_sound_heap_local_gen _ landed_fixes_sound). Qed.

Theorem rewrite_symbol_type_sound_heap_local :
  rewrite_symbol_type_sound_heap_local_statement.
Proof.
  exact (rewrite_symbol_type_sound_heap_local_gen _ all_fixes_sound).
Qed.

Theorem rewrite_symbol_type_sound_heap_local_landed :
  rewrite_symbol_type_sound_heap_local_landed_statement.
Proof.
  exact (rewrite_symbol_type_sound_heap_local_gen _ landed_fixes_sound).
Qed.

Theorem rewrite_result_type_sound_heap_local :
  rewrite_result_type_sound_heap_local_statement.
Proof.
  exact (rewrite_result_type_sound_heap_local_gen _ all_fixes_sound).
Qed.

Theorem rewrite_result_type_sound_heap_local_landed :
  rewrite_result_type_sound_heap_local_landed_statement.
Proof.
  exact (rewrite_result_type_sound_heap_local_gen _ landed_fixes_sound).
Qed.

Theorem rewrite_unaliased_type_sound_heap_local :
  rewrite_unaliased_type_sound_heap_local_statement.
Proof.
  exact (rewrite_unaliased_type_sound_heap_local_gen _ all_fixes_sound).
Qed.

Theorem rewrite_unaliased_type_sound_heap_local_landed :
  rewrite_unaliased_type_sound_heap_local_landed_statement.
Proof.
  exact (rewrite_unaliased_type_sound_heap_local_gen _ landed_fixes_sound).
Qed.

Theorem unaliased_symbol_type_sound_heap_local :
  unaliased_symbol_type_sound_heap_local_statement.
Proof.
  exact (rewrite_unaliased_symbol_type_sound_heap_local_gen _
           all_fixes_sound).
Qed.

Theorem unaliased_symbol_type_sound_heap_local_landed :
  unaliased_symbol_type_sound_heap_local_landed_statement.
Proof.
  exact (rewrite_unaliased_symbol_type_sound_heap_local_gen _
           landed_fixes_sound).
Qed.

Theorem unaliased_result_type_sound_heap_local :
  unaliased_result_type_sound_heap_local_statement.
Proof.
  exact (rewrite_unaliased_result_type_sound_heap_local_gen _
           all_fixes_sound).
Qed.

Theorem unaliased_result_type_sound_heap_local_landed :
  unaliased_result_type_sound_heap_local_landed_statement.
Proof.
  exact (rewrite_unaliased_result_type_sound_heap_local_gen _
           landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* From the relation of the simulation to [statics_rebuilt_heap]            *)
(* ======================================================================== *)

Section Exit_heap.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hexact : field_used_exact is_local G sol.
Variable kept : value_slot -> bool.
Variable called : node -> code_id -> Prop.
Variable exact partials : Prop.
Variable located : location -> option node.
Variable β : location -> location -> Prop.

Local Abbreviation rel :=
  (RBP.rel G sol kept called exact partials located β True).

(* A value of the first run, related by the simulation at [n] to one of
   the second, is rebuilt as [rebuilt_value_heap] says, if [n] has any
   source where the value is a constant. *)
Lemma rel_rebuilt_value_heap :
  ∀ (v : value) (n : node) (v' : value),
    rel v n v' ->
    (∀ c : const, v = Value_const c -> s_any_source sol n = true) ->
    rebuilt_value_heap G sol β v n v'.
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ (n : node) (v' : value),
              rel v n v' ->
              (∀ c : const, v = Value_const c ->
               s_any_source sol n = true) ->
              rebuilt_value_heap G sol β v n v'));
    cbn [RBP.rel rebuilt_value_heap].
  - intros c n v' [_ ->] Has. split; [ exact (Has c eq_refl) | reflexivity ].
  - intros code slots _ n v' [].
  - intros slot decls vs IH n v' (sites & vs' & Hmay & -> & _ & Hse & Hall)
      _.
    exists sites, vs'. split; [ reflexivity | split; [ exact Hmay | ] ].
    apply (closure_rebuilt is_local G sol Hc Hexact kept called exact partials
             located β (rebuilt_value_heap G sol β) decls sites vs vs'
             (Hse I)).
    + intros f id Hf. destruct (Hall f id Hf) as (_ & _ & Hfs & Hsr).
      exact (conj Hfs Hsr).
    + intros w x from x' Hl Hrel Has.
      rewrite Forall_forall in IH.
      exact (IH (w, x) (lookup_in _ _ _ Hl) from x' Hrel Has).
  - intros s n v' (-> & Hmay & Hhu) _.
    split; [ reflexivity | split; [ exact Hmay | exact Hhu ] ].
  - intros tag fields IH n v' (a & fields' & Hmay & -> & Hf & Hhs & Hk) _.
    exists a, fields'.
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    revert fields' Hf. generalize 0 as i.
    induction IH as [ | field fields Hfield _ IHf]; intros i [ | field' fields']
      Hf; cbn [RBP.fields_related fields_rebuilt] in Hf |- *;
      try contradiction; [ exact I | ].
    apply Forall_cons_iff in Hk as [Hkf Hk].
    destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
    split.
    + rewrite Hkf. exists from.
      split; [ exact Hcons | split ].
      * intros Hu.
        split; [ exact (used_field_hu is_local G sol Hc Hexact _ _ _ Hu Hcons)
               | ].
        apply (Hfield from field' (Hused Hu)).
        intros c ->. pose proof (Hused Hu) as Hr. cbn [RBP.rel] in Hr.
        cbn [kind_of_value] in Hkf. rewrite Hkf in Hr.
        exact (proj1 Hr eq_refl).
      * exact Hunused.
    + exact (IHf Hk (S i) fields' Hrest).
  - intros elements _ n v' (_ & elements' & -> & _) _.
    exists elements'. reflexivity.
  - intros c n v' (a & bn & from & Hmay & Hhs & Hk & Hcons & ->) _.
    exists a, bn, from. auto.
  - intros l n v' (a & l' & _ & _ & -> & Hb) _.
    exists l'. split; [ reflexivity | exact Hb ].
  - intros callee args arity _ _ n v' [_ (m & closure' & args' & -> & _)] _.
    exists closure', args'. reflexivity.
Qed.

End Exit_heap.

(* The static environments that the simulation relates, with the heap
   ([HE.exit_statics_rel_heap]), are related by [statics_rebuilt_heap]
   for the correspondence of the simulation, if the static values of the
   symbols with a use of the first are of the fragment. *)
Lemma statics_rel_rebuilt_heap_used (is_local : field -> bool) (G : graph)
    (sol : solution) (keep : value_slot -> bool) (L : location -> option node)
    (statics statics' : static_env) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  HE.exit_statics_rel_heap G sol keep L statics statics' ->
  statics_of_used_values (fun s => s_has_usage sol (Node_symbol s))
    statics ->
  statics_rebuilt_heap G sol (RBP.kept_beta sol True L) statics statics'.
Proof.
  intros Hc Hexact (cds & F & b & ids & ff & cc & cdeps & cmds & ty
                    & Hnone & Hblocks & Honly & _)
    Hvals s allocated Hl Hhu.
  pose proof (Hvals s allocated Hl Hhu) as Hval.
  destruct (Honly s allocated Hl)
    as [(tag & fields & ->)
       | [(slot & syms & decls & vs & -> & Htarget & Hbound)
         | (Ho & Hbound)]].
  - destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | ].
    clear Hl Hl' Hblocks Honly. revert fields' Hf. generalize 0 as i.
    induction fields as [ | field fields IH]; intros i [ | field' fields']
      Hf; cbn [RBP.fields_related fields_rebuilt] in Hf |- *;
      try contradiction; [ exact I | ].
    destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
    assert (Hkf : kind_of_value field = Value)
      by exact (Hval field (or_introl eq_refl)).
    split.
    + unfold field_rebuilt. rewrite Hkf. exists from.
      split; [ exact Hcons | split; [ | exact Hunused ] ].
      intros Hu.
      split; [ exact (used_field_hu is_local G sol Hc Hexact _ _ _ Hu Hcons)
             | ].
      apply (rel_rebuilt_value_heap is_local G sol Hc Hexact _ _ _ _ _ _
               field from field' (Hused Hu)).
      intros c ->. pose proof (Hused Hu) as Hr. cbn [RBP.rel] in Hr.
      cbn [kind_of_value] in Hkf. rewrite Hkf in Hr.
      exact (proj1 Hr eq_refl).
    + exact (IH (fun v Hv => Hval v (or_intror Hv)) (S i) fields' Hrest).
  - destruct Htarget as [Hn | (vs' & Hls' & _ & _ & Hsites)];
      [ destruct (Hbound Hhu Hn) | ].
    destruct (Hsites I) as (sites & Hsite & Hse & Hall).
    exists sites, vs'.
    split; [ exact Hls' | split; [ exact Hsite | ] ].
    apply (closure_rebuilt is_local G sol Hc Hexact _ _ _ _ _ _
             (rebuilt_value_heap G sol (RBP.kept_beta sol True L)) decls
             sites vs vs' Hse Hall).
    intros w x from x' Hlw Hrel Has.
    exact (rel_rebuilt_value_heap is_local G sol Hc Hexact _ _ _ _ _ _ x from
             x' Hrel Has).
  - destruct allocated as [ | | | contents | bytes | elements | ];
      cbn [static_rebuilt_heap]; cbn [RBP.static_other_rel] in Ho;
      try contradiction.
    + destruct Ho as [Hn | (v' & Hls' & Hv')];
        [ destruct (Hbound Hhu Hn) | ].
      rewrite Hls'. f_equal. f_equal.
      destruct Hv' as [Hv' | Hsim]; [ exact Hv' | ].
      destruct contents as [c | | | | | | | | ];
        cbn [kind_of_value boxable_number_of_kind] in Hval;
        try (exfalso; exact (Hval eq_refl)).
      exact Hsim.
    + destruct Ho as [Hn | Hls']; [ destruct (Hbound Hhu Hn) | exact Hls' ].
    + destruct Ho as [Hn | (elements' & Hls' & _)];
        [ destruct (Hbound Hhu Hn) | ].
      exists elements'. exact Hls'.
Qed.

(* The same, if the static values of the first are of the fragment where
   the unit exits. *)
Lemma statics_rel_rebuilt_heap (is_local : field -> bool) (G : graph)
    (sol : solution) (keep : value_slot -> bool) (L : location -> option node)
    (statics statics' : static_env) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  HE.exit_statics_rel_heap G sol keep L statics statics' ->
  statics_of_values statics ->
  statics_rebuilt_heap G sol (RBP.kept_beta sol True L) statics statics'.
Proof.
  intros Hc Hexact Hrel Hvals.
  apply (statics_rel_rebuilt_heap_used is_local G sol keep L statics statics'
           Hc Hexact Hrel).
  intros s allocated Hl _. exact (Hvals s allocated Hl).
Qed.

(* The static sets kept whole, as [statics_rel_whole] has them. *)
Lemma statics_rel_whole_heap (G : graph) (sol : solution)
    (keep : value_slot -> bool) (L : location -> option node)
    (statics statics' : static_env) :
  HE.exit_statics_rel_heap G sol keep L statics statics' ->
  FlambdaClosuresSpec.static_sets_bound statics ->
  FlambdaClosuresSpec.static_sets_bound statics' ->
  static_sets_whole sol statics statics'.
Proof.
  intros (cds & F & b & ids & ff & cc & cdeps & cmds & ty & _ & _ & Honly
          & _)
    Hbound Hbound' s slot syms decls vs Hl Hhu.
  destruct (Hbound s slot syms decls vs Hl) as [Hslot Hset].
  split; [ exact Hslot | ].
  destruct (Honly s _ Hl)
    as [(tag & fields & Heq)
       | [(slot0 & syms0 & decls0 & vs0 & Heq & Htarget & Hbinds)
         | (Ho & _)]];
    [ discriminate Heq | | destruct Ho ].
  injection Heq as <- <- <- <-.
  destruct Htarget as [Hn | (vs' & Hls' & _)]; [ destruct (Hbinds Hhu Hn) | ].
  exists vs'. intros g s_g Hg.
  split; [ exact (Hset g s_g (lookup_in _ _ _ Hg)) | ].
  exact (proj2 (Hbound' s slot syms decls vs' Hls') g s_g
           (lookup_in _ _ _ Hg)).
Qed.

(* [CE.closures_exact_of_bound_once], of the units with the heap. *)
Lemma closures_exact_of_bound_once_heap (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  in_heap_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  RBP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e.
Proof.
  intros Hfrag Hwf Honce.
  assert (Hold : CE.no_old_sets e).
  { unfold in_heap_fragment in Hfrag. apply andb_true_iff in Hfrag as [Hf _].
    exact (proj1 (CE.fragment_no_old_sets _ _ _) e Hf). }
  assert (Hnl : NoDup (map fst (let_bindings e))).
  { unfold variables_bound_once in Honce. apply distinct_NoDup in Honce.
    exact (NoDup_app_remove_r _ _ Honce). }
  assert (Hns : NoDup (map fst (CE.sym_bindings e))).
  { rewrite (proj1 CE.sym_bindings_static_names e Hold).
    apply distinct_NoDup. unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  assert (Hsets : sets_well_formed e = true).
  { unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  destruct (CE.traverse_unit_groups flag u e Hnl Hns Hold)
    as (Gs & a & -> & Hinv & Hpres).
  apply (proj1 (CE.closures_exact_struct _ _) e).
  - intros x d Hin. destruct d as [ | | | slot decls vs | | | ];
      cbn [RBP.named_exact]; try exact I.
    exact (CE.closure_node_exact e Hnl Gs a Hinv x slot decls vs Hin).
  - intros t c Hin. destruct c as [ | slot cs decls vs | | | | ];
      cbn [RBP.binding_exact]; try exact I.
    destruct (proj1 CE.static_sets_whole e Hsets t slot cs decls vs Hin)
      as [Hdom Hwhole].
    split.
    + exact (CE.static_node_exact e Hns Gs a Hinv t slot cs decls vs Hin
               Hdom Hwhole Hpres).
    + intros g. rewrite !is_bound_among. f_equal. exact Hdom.
Qed.

(* ======================================================================== *)
(* The exit relation with the heap                                          *)
(* ======================================================================== *)

Lemma kept_beta_one_to_one (sol : solution) (L : location -> option node) :
  locations_one_to_one (RBP.kept_beta sol True L).
Proof.
  split.
  - intros l l1 l2 H1 H2. exact (RBP.kept_beta_fun sol True L l l1 l2 H1 H2).
  - intros l1 l2 l' H1 H2.
    exact (RBP.kept_beta_inj sol True L l1 l2 l' H1 H2).
Qed.

(* Every symbol bound in a run of the fragment with the heap is an
   allocation of its own (FlambdaReaperFlowSpec.flow_sound_local_statement,
   which is of that fragment). *)
Lemma statics_allocated_heap_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (n : nat) (s : state) :
  is_local_slots_only is_local -> in_heap_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
  statics_allocated (solution_of_tables T) (state_statics s).
Proof.
  intros Hlocal Hfrag Hcheck Hlrs Hrun.
  destruct (FlambdaReaperFlowProofs.flow_sound_local flag is_local u e T n
              Hlocal Hfrag Hcheck Hlrs)
    as (located & Hflows).
  rewrite Hrun in Hflows. destruct Hflows as (_ & _ & Hst & _).
  intros x allocated Hl. exact (proj1 (Hst x allocated Hl)).
Qed.

Lemma exit_relation_whole_heap_of_exact_local
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  RBP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e ->
  exits_of_values e ->
  rebuild_exit_relation_whole_heap (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Hxe Hvals
    n s k vs hp rd Hrun Hstep.
  split; [ exact (statics_allocated_heap_local flag is_local u e T n s Hlocal
                    Hfrag Hcheck Hlrs Hrun) | ].
  destruct (HE.rebuild_exit_statics_heap_local flag is_local u e T
              preserve_direct_calls Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
              Hlrs Hcheck_exact Hxe n s k vs hp rd Hrun Hstep)
    as (L & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & Hst).
  exists (RBP.kept_beta (solution_of_tables T) True L), m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  split; [ apply kept_beta_one_to_one | split ].
  - exact (statics_rel_rebuilt_heap is_local _ _ _ _ _ _
             (check_closed_sound is_local _ _ Hcheck)
             (check_field_used_exact_sound is_local _ _ Hcheck_exact) Hst
             (Hvals n s k vs hp rd Hrun Hstep)).
  - apply (statics_rel_whole_heap _ _ _ _ _ _ Hst).
    + exact (FlambdaClosuresProofs.well_formed_sets_bound u e n s Hwf Hrun).
    + exact (rebuilt_sets_bound _ T preserve_direct_calls u e m t Hwf Hm).
Qed.

Lemma exit_relation_heap_of_whole (G : graph) (sol : solution) (e e' : expr) :
  rebuild_exit_relation_whole_heap G sol e e' ->
  rebuild_exit_relation_heap G sol e e'.
Proof.
  intros H n s k vs hp rd Hrun Hstep.
  destruct (H n s k vs hp rd Hrun Hstep)
    as [Ha (β & m & t & vs' & hp' & rd' & H1 & H2 & H3 & H4 & H5 & _)].
  split; [ exact Ha | ].
  exists β, m, t, vs', hp', rd'. repeat (split; [ assumption | ]).
  assumption.
Qed.

Theorem rebuild_exit_relation_whole_heap_holds_local :
  rebuild_exit_relation_whole_heap_holds_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (exit_relation_whole_heap_of_exact_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact
           (closures_exact_of_bound_once_heap flag u e Hfrag Hwf Honce)
           Hvals).
Qed.

Theorem rebuild_exit_relation_whole_heap_holds :
  rebuild_exit_relation_whole_heap_holds_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  exact (rebuild_exit_relation_whole_heap_holds_local flag is_local u e T b
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem rebuild_exit_relation_heap_holds_local :
  rebuild_exit_relation_heap_holds_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (exit_relation_heap_of_whole _ _ _ _
           (rebuild_exit_relation_whole_heap_holds_local flag is_local u e T b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
              Hvals)).
Qed.

Theorem rebuild_exit_relation_heap_holds :
  rebuild_exit_relation_heap_holds_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact Honce Hvals.
  exact (exit_relation_heap_of_whole _ _ _ _
           (rebuild_exit_relation_whole_heap_holds flag is_local u e T b
              Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck Hcheck_exact Honce
              Hvals)).
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Lemma exported_types_sound_heap_gen (fx : fixes) :
  fixes_sound fx -> exported_types_sound_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun Hstep Henv Hfrag
    Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & _ & Hst)].
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  exact (rewrite_symbol_type_sound_heap_gen fx false Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' β sym allocated (or_introl Hleast) Hlocal Hsets Henv
           Hfrag Halloc Hst (fun H => match diff_false_true H with end)
           Hcopies Hl).
Qed.

Lemma exported_unaliased_types_sound_heap_gen (fx : fixes) :
  fixes_sound fx -> exported_unaliased_types_sound_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun Hstep Henv Hfrag
    Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & _ & Hst
                & Hwhole)].
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  exact (rewrite_symbol_type_sound_heap_gen fx true Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' β sym allocated (or_introl Hleast) Hlocal Hsets Henv
           Hfrag Halloc Hst (fun _ => Hwhole) Hcopies Hl).
Qed.

Theorem exported_types_sound_heap : exported_types_sound_heap_statement.
Proof. exact (exported_types_sound_heap_gen _ all_fixes_sound). Qed.

Theorem exported_types_sound_heap_landed :
  exported_types_sound_heap_landed_statement.
Proof. exact (exported_types_sound_heap_gen _ landed_fixes_sound). Qed.

Theorem exported_unaliased_types_sound_heap :
  exported_unaliased_types_sound_heap_statement.
Proof. exact (exported_unaliased_types_sound_heap_gen _ all_fixes_sound). Qed.

Theorem exported_unaliased_types_sound_heap_landed :
  exported_unaliased_types_sound_heap_landed_statement.
Proof.
  exact (exported_unaliased_types_sound_heap_gen _ landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* With no premise on the runs                                              *)
(* ======================================================================== *)

Lemma exported_types_sound_end_to_end_heap_local_gen (fx : fixes) :
  fixes_sound fx -> exported_types_sound_end_to_end_heap_local_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  destruct (rebuild_exit_relation_whole_heap_holds_local flag is_local u e T b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
              (least_field_used_exact_local is_local _ T Hleast) Honce Hvals
              n s k vs hp rd Hrun Hstep)
    as [Halloc (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & _ & Hst & _)].
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  exact (rewrite_symbol_type_sound_heap_gen fx false Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' β sym allocated (or_intror Hleast) Hlocal Hsets Henv Hfr
           Halloc Hst (fun H => match diff_false_true H with end) Hcopies Hl).
Qed.

Lemma unaliased_types_end_to_end_heap_local_gen (fx : fixes) :
  fixes_sound fx -> unaliased_types_end_to_end_heap_local_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  destruct (rebuild_exit_relation_whole_heap_holds_local flag is_local u e T b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
              (least_field_used_exact_local is_local _ T Hleast) Honce Hvals
              n s k vs hp rd Hrun Hstep)
    as [Halloc (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & _ & Hst
                & Hwhole)].
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  exact (rewrite_symbol_type_sound_heap_gen fx true Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' β sym allocated (or_intror Hleast) Hlocal Hsets Henv Hfr
           Halloc Hst (fun _ => Hwhole) Hcopies Hl).
Qed.

Lemma exported_types_sound_end_to_end_heap_gen (fx : fixes) :
  fixes_sound fx -> exported_types_sound_end_to_end_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (exported_types_sound_end_to_end_heap_local_gen fx Hfx flag is_local
           u e T b sets tenv canon valuation valuation' n s k vs hp rd sym
           allocated Hlocal Hfrag Hdc Hrk Hwf Hexits
           (least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Lemma unaliased_types_end_to_end_heap_gen (fx : fixes) :
  fixes_sound fx -> unaliased_types_end_to_end_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (unaliased_types_end_to_end_heap_local_gen fx Hfx flag is_local
           u e T b sets tenv canon valuation valuation' n s k vs hp rd sym
           allocated Hlocal Hfrag Hdc Hrk Hwf Hexits
           (least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Theorem exported_types_sound_end_to_end_heap :
  exported_types_sound_end_to_end_heap_statement.
Proof. exact (exported_types_sound_end_to_end_heap_gen _ all_fixes_sound). Qed.

Theorem exported_types_sound_end_to_end_heap_landed :
  exported_types_sound_end_to_end_heap_landed_statement.
Proof.
  exact (exported_types_sound_end_to_end_heap_gen _ landed_fixes_sound).
Qed.

Theorem unaliased_types_end_to_end_heap :
  unaliased_types_end_to_end_heap_statement.
Proof. exact (unaliased_types_end_to_end_heap_gen _ all_fixes_sound). Qed.

Theorem unaliased_types_end_to_end_heap_landed :
  unaliased_types_end_to_end_heap_landed_statement.
Proof. exact (unaliased_types_end_to_end_heap_gen _ landed_fixes_sound). Qed.

Theorem exported_types_sound_end_to_end_heap_local :
  exported_types_sound_end_to_end_heap_local_statement.
Proof.
  exact (exported_types_sound_end_to_end_heap_local_gen _ all_fixes_sound).
Qed.

Theorem exported_types_sound_end_to_end_heap_local_landed :
  exported_types_sound_end_to_end_heap_local_landed_statement.
Proof.
  exact (exported_types_sound_end_to_end_heap_local_gen _ landed_fixes_sound).
Qed.

Theorem unaliased_types_end_to_end_heap_local :
  unaliased_types_end_to_end_heap_local_statement.
Proof.
  exact (unaliased_types_end_to_end_heap_local_gen _ all_fixes_sound).
Qed.

Theorem unaliased_types_end_to_end_heap_local_landed :
  unaliased_types_end_to_end_heap_local_landed_statement.
Proof.
  exact (unaliased_types_end_to_end_heap_local_gen _ landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* With the check of the term                                               *)
(* ======================================================================== *)

Lemma types_end_to_end_term_heap_gen (fx : fixes) :
  fixes_sound fx -> types_end_to_end_term_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast
    Honce Hvals.
  exact (exported_types_sound_end_to_end_heap_gen fx Hfx flag is_local u e T
           b sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (FlambdaReaperTypesTermProofs.unit_of_values_exits e Hvals)).
Qed.

Lemma unaliased_types_end_to_end_term_heap_gen (fx : fixes) :
  fixes_sound fx -> unaliased_types_end_to_end_term_heap_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast
    Honce Hvals.
  exact (unaliased_types_end_to_end_heap_gen fx Hfx flag is_local u e T
           b sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (FlambdaReaperTypesTermProofs.unit_of_values_exits e Hvals)).
Qed.

Lemma types_end_to_end_term_heap_local_gen (fx : fixes) :
  fixes_sound fx -> types_end_to_end_term_heap_local_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (exported_types_sound_end_to_end_heap_local_gen fx Hfx flag is_local
           u e T b sets tenv canon valuation valuation' n s k vs hp rd sym
           allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (FlambdaReaperTypesTermProofs.unit_of_values_exits e Hvals)).
Qed.

Lemma unaliased_types_end_to_end_term_heap_local_gen (fx : fixes) :
  fixes_sound fx -> unaliased_types_end_to_end_term_heap_local_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    k vs hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals.
  exact (unaliased_types_end_to_end_heap_local_gen fx Hfx flag is_local
           u e T b sets tenv canon valuation valuation' n s k vs hp rd sym
           allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce
           (FlambdaReaperTypesTermProofs.unit_of_values_exits e Hvals)).
Qed.

Theorem types_end_to_end_term_heap : types_end_to_end_term_heap_statement.
Proof. exact (types_end_to_end_term_heap_gen _ all_fixes_sound). Qed.

Theorem types_end_to_end_term_heap_landed :
  types_end_to_end_term_heap_landed_statement.
Proof. exact (types_end_to_end_term_heap_gen _ landed_fixes_sound). Qed.

Theorem unaliased_types_end_to_end_term_heap :
  unaliased_types_end_to_end_term_heap_statement.
Proof. exact (unaliased_types_end_to_end_term_heap_gen _ all_fixes_sound). Qed.

Theorem unaliased_types_end_to_end_term_heap_landed :
  unaliased_types_end_to_end_term_heap_landed_statement.
Proof.
  exact (unaliased_types_end_to_end_term_heap_gen _ landed_fixes_sound).
Qed.

Theorem types_end_to_end_term_heap_local :
  types_end_to_end_term_heap_local_statement.
Proof. exact (types_end_to_end_term_heap_local_gen _ all_fixes_sound). Qed.

Theorem types_end_to_end_term_heap_local_landed :
  types_end_to_end_term_heap_local_landed_statement.
Proof. exact (types_end_to_end_term_heap_local_gen _ landed_fixes_sound). Qed.

Theorem unaliased_types_end_to_end_term_heap_local :
  unaliased_types_end_to_end_term_heap_local_statement.
Proof.
  exact (unaliased_types_end_to_end_term_heap_local_gen _ all_fixes_sound).
Qed.

Theorem unaliased_types_end_to_end_term_heap_local_landed :
  unaliased_types_end_to_end_term_heap_local_landed_statement.
Proof.
  exact (unaliased_types_end_to_end_term_heap_local_gen _
           landed_fixes_sound).
Qed.
