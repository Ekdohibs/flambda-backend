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


(* Proofs of the statements of FlambdaJoinTypesKeepsSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaJoinEnv FlambdaJoinEnvSpec
  FlambdaJoinEnvProofs FlambdaJoinTypes FlambdaJoinTypesProofs
  FlambdaJoinTypesKeepsSpec.
Import ListNotations.

Section Keeps.

Context {state : Type} (keeps : state -> state -> Prop).

Hypothesis keeps_refl : ∀ s : state, keeps s s.
Hypothesis keeps_trans :
  ∀ s1 s2 s3 : state, keeps s1 s2 -> keeps s2 s3 -> keeps s1 s3.

Context
  (joined_env : state -> Index.t -> option Typing_env.t)
  (n_way_join_simples :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state))
  (import_type : state -> flambda_type -> option state)
  (join_env_n_way_join_env_extension :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state)))
  (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
  (reduce_head_of_kind_naked_immediate :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate)
  (type_eqb : flambda_type -> flambda_type -> bool)
  (block_shape_eqb : block_shape -> block_shape -> bool).

Hypothesis Hsimples :
  ∀ (s s' : state) (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    n_way_join_simples s kind simples = Some (result, s') -> keeps s s'.
Hypothesis Himport :
  ∀ (s s' : state) (ty : flambda_type),
    import_type s ty = Some s' -> keeps s s'.
Hypothesis Hextension :
  ∀ (s s' : state) (extensions : list (Index.t * env_extension))
    (joined : env_extension),
    join_env_n_way_join_env_extension s extensions
    = Some (Or_bottom.Ok (joined, s')) ->
    keeps s s'.

(* A function that threads the state keeps the relation; and one that
   may answer bottom. *)
Definition step_keeps {A B : Type}
    (step : state -> A -> option (B * state)) : Prop :=
  ∀ (s s' : state) (a : A) (b : B), step s a = Some (b, s') -> keeps s s'.

Definition join_keeps {A B : Type}
    (join : state -> A -> option (Or_bottom.t (B * state))) : Prop :=
  ∀ (s s' : state) (a : A) (b : B),
    join s a = Some (Or_bottom.Ok (b, s')) -> keeps s s'.

(* What is left of an equation once the calls that change the state
   have been named: the matches on what does not, and an answer whose
   state is one already related. *)
Local Ltac destruct_innermost x :=
  lazymatch x with
  | context [match ?y with _ => _ end] => destruct_innermost y
  | _ => destruct x
  end.

Local Ltac finish Hstep :=
  repeat
    match type of Hstep with
    | context [match ?x with _ => _ end] =>
      destruct_innermost x; try discriminate Hstep
    end;
  try discriminate Hstep;
  first [ injection Hstep as <- <- | injection Hstep as <- ];
  solve [ eauto ].

Lemma n_way_join_unknown_keeps {A B : Type}
    (contents :
       state -> list (Index.t * A) -> option (Or_unknown.t B * state)) :
  step_keeps contents -> step_keeps (n_way_join_unknown contents).
Proof.
  intros Hcontents s s' a b Hstep. unfold n_way_join_unknown in Hstep.
  destruct (known_contents [] a) as [known | ];
    [ exact (Hcontents _ _ _ _ Hstep) | finish Hstep ].
Qed.

Lemma n_way_join_relation_simples_keeps :
  step_keeps (n_way_join_relation_simples n_way_join_simples).
Proof.
  intros s s' a b Hstep. unfold n_way_join_relation_simples in Hstep.
  destruct a as [simples | ]; [ | finish Hstep ].
  destruct (n_way_join_simples s (Naked_number Kind_naked_immediate) simples)
    as [[joined s1] | ] eqn:Hjoin; [ | discriminate ].
  apply Hsimples in Hjoin. finish Hstep.
Qed.

Lemma n_way_join_is_null_keeps :
  step_keeps (n_way_join_is_null n_way_join_simples).
Proof.
  intros s s' heads b Hstep. unfold n_way_join_is_null in Hstep.
  destruct (is_null_simples heads) as [simples | ]; [ | finish Hstep ].
  destruct (n_way_join_simples s (Naked_number Kind_naked_immediate) simples)
    as [[joined s1] | ] eqn:Hjoin; [ | discriminate ].
  apply Hsimples in Hjoin. finish Hstep.
Qed.

Lemma n_way_join_head_of_kind_naked_immediate_keeps :
  step_keeps
    (n_way_join_head_of_kind_naked_immediate joined_env
       reduce_head_of_kind_naked_immediate).
Proof.
  intros s s' heads b Hstep.
  unfold n_way_join_head_of_kind_naked_immediate in Hstep. finish Hstep.
Qed.

Lemma n_way_join_env_extension_keeps :
  step_keeps (n_way_join_env_extension join_env_n_way_join_env_extension).
Proof.
  intros s s' extensions b Hstep. unfold n_way_join_env_extension in Hstep.
  destruct (join_env_n_way_join_env_extension s extensions)
    as [[[joined s1] | ] | ] eqn:Hjoin; [ | | discriminate ].
  - apply Hextension in Hjoin. finish Hstep.
  - finish Hstep.
Qed.

Lemma generic_merge_map_known_keeps {K value other : Type}
    (eqb : K -> K -> bool) (Heqb : decides_equality eqb)
    (of_other : other -> value)
    (join :
       state -> list (Index.t * value)
       -> option (Or_bottom.t (value * state))) :
  join_keeps join -> step_keeps (generic_merge_map_known eqb of_other join).
Proof.
  intros Hjoin s s' maps joined Hstep.
  unfold generic_merge_map_known in Hstep.
  destruct maps as [ | first rest]; [ finish Hstep | ].
  eapply (filter_map_fold_spec eqb Heqb (fun _ => True) keeps keeps_refl
            keeps_trans)
    in Hstep; [ exact (proj1 (proj2 Hstep)) | | exact I ].
  intros key values s0 a s0' _ _ Hcall.
  split; [ exact I | ].
  destruct (join s0 values) as [[[value' s1] | ] | ] eqn:Hjoined;
    [ apply Hjoin in Hjoined | | discriminate ]; finish Hcall.
Qed.

(* ------------------------------------------------------------------------ *)
(* Cases of row-like structures                                             *)
(* ------------------------------------------------------------------------ *)

Section Cases.

Context {lattice shape maps_to : Type}
  (n_way_join_maps_to :
     state -> shape -> list (Index.t * maps_to) -> option (maps_to * state))
  (equal_index : lattice -> lattice -> bool)
  (inter_index : lattice -> lattice -> lattice)
  (n_way_join_shape : list shape -> or_unknown_or_bottom shape).

Hypothesis Hmaps_to :
  ∀ sh : shape, step_keeps (fun s args => n_way_join_maps_to s sh args).

Lemma n_way_join_case_keeps :
  join_keeps
    (n_way_join_case join_env_n_way_join_env_extension n_way_join_maps_to
       equal_index inter_index n_way_join_shape).
Proof.
  intros s s' cases b Hstep. unfold n_way_join_case in Hstep.
  destruct (n_way_join_index _ _ _ _) as [ | index | ];
    [ finish Hstep | | discriminate ].
  match type of Hstep with
  | context [n_way_join_maps_to s ?sh ?args] =>
    destruct (n_way_join_maps_to s sh args) as [[joined s1] | ] eqn:Hmaps;
      [ | discriminate ]
  end.
  apply (Hmaps_to (index_shape index)) in Hmaps.
  match type of Hstep with
  | context [n_way_join_env_extension ?f s1 ?args] =>
    destruct (n_way_join_env_extension f s1 args) as [[extension s2] | ]
      eqn:Hext; [ | discriminate ]
  end.
  apply n_way_join_env_extension_keeps in Hext. finish Hstep.
Qed.

Lemma n_way_join_knowns_keeps :
  join_keeps
    (n_way_join_knowns join_env_n_way_join_env_extension n_way_join_maps_to
       equal_index inter_index n_way_join_shape).
Proof.
  intros s s' cases b Hstep. unfold n_way_join_knowns in Hstep.
  destruct (all_known cases) as [known | ];
    [ exact (n_way_join_case_keeps _ _ _ _ Hstep) | finish Hstep ].
Qed.

Lemma n_way_join_row_like_keeps {known : Type}
    (merge_map_known :
       (state
        -> list
             (Index.t
              * Or_unknown.t
                  (row_like_case lattice shape maps_to env_extension))
        -> option
             (Or_bottom.t
                (Or_unknown.t
                   (row_like_case lattice shape maps_to env_extension)
                 * state)))
       -> state
       -> list
            (Index.t
             * (known
                * Or_bottom.t
                    (row_like_case lattice shape maps_to env_extension)))
       -> option (known * state)) :
  (∀ join, join_keeps join -> step_keeps (merge_map_known join)) ->
  ∀ (s s' : state)
    (known_cases :
       list
         (Index.t
          * (known
             * Or_bottom.t
                 (row_like_case lattice shape maps_to env_extension))))
    (other :
       list
         (Index.t
          * Or_bottom.t (row_like_case lattice shape maps_to env_extension)))
    (answer :
       Or_unknown.t
         (known
          * Or_bottom.t (row_like_case lattice shape maps_to env_extension))),
    n_way_join_row_like join_env_n_way_join_env_extension n_way_join_maps_to
      equal_index inter_index n_way_join_shape merge_map_known s known_cases
      other
    = Some (answer, s') ->
    keeps s s'.
Proof.
  intros Hmerge s s' known_cases other answer Hstep.
  unfold n_way_join_row_like in Hstep.
  destruct (merge_map_known _ s known_cases) as [[joined s1] | ] eqn:Hknown;
    [ | discriminate ].
  apply (Hmerge _ n_way_join_knowns_keeps) in Hknown.
  match type of Hstep with
  | context [n_way_join_case ?a ?b ?c ?d ?e s1 ?l] =>
    destruct (n_way_join_case a b c d e s1 l) as [[[joined_other s2] | ] | ]
      eqn:Hother; [ apply n_way_join_case_keeps in Hother | | discriminate ]
  end; finish Hstep.
Qed.

End Cases.

(* ------------------------------------------------------------------------ *)
(* The functions that call the join of types                                *)
(* ------------------------------------------------------------------------ *)

Section Body.

Variable n_way_join_rec :
  state -> list (Index.t * flambda_type)
  -> option (Or_unknown.t flambda_type * state).

Hypothesis Hrec : step_keeps n_way_join_rec.

Lemma n_way_join_generic_product_keeps :
  step_keeps (n_way_join_generic_product n_way_join_rec).
Proof.
  intros s s' products joined Hstep.
  unfold n_way_join_generic_product in Hstep.
  destruct products as [ | first rest]; [ finish Hstep | ].
  eapply (filter_map_fold_spec Pos.eqb pos_eqb_decides_equality
            (fun _ => True) keeps keeps_refl keeps_trans)
    in Hstep; [ exact (proj1 (proj2 Hstep)) | | exact I ].
  intros key tys s0 a s0' _ _ Hcall.
  split; [ exact I | ].
  destruct (n_way_join_rec s0 tys) as [[joined_ty s1] | ] eqn:Hjoined;
    [ | discriminate ].
  apply Hrec in Hjoined. finish Hcall.
Qed.

Lemma import_types_keeps (tys : list flambda_type) :
  ∀ s s' : state, import_types import_type s tys = Some s' -> keeps s s'.
Proof.
  induction tys as [ | ty tys IH]; intros s s' Hstep; simpl in Hstep.
  - injection Hstep as <-. apply keeps_refl.
  - destruct (import_type s ty) as [s1 | ] eqn:Himported; [ | discriminate ].
    exact (keeps_trans _ _ _ (Himport _ _ _ Himported) (IH _ _ Hstep)).
Qed.

Lemma join_fields_keeps (sh : block_shape)
    (fields : list (Index.t * list flambda_type)) (count : nat) :
  ∀ (index : nat) (s s' : state) (answer : list flambda_type),
    join_fields n_way_join_rec s sh fields index count = Some (answer, s') ->
    keeps s s'.
Proof.
  induction count as [ | count IH]; intros index s s' answer Hstep;
    simpl in Hstep.
  - injection Hstep as <- <-. apply keeps_refl.
  - destruct (fields_at index fields) as [tys | ]; [ | discriminate ].
    destruct (n_way_join_rec s tys) as [[joined s1] | ] eqn:Hjoined;
      [ | discriminate ].
    apply Hrec in Hjoined.
    match type of Hstep with
    | context [join_fields ?f s1 ?a ?b ?c ?d] =>
      destruct (join_fields f s1 a b c d) as [[rest s2] | ] eqn:Hrest
    end.
    + apply IH in Hrest. finish Hstep.
    + finish Hstep.
Qed.

Lemma n_way_join_int_indexed_product_keeps (sh : block_shape) :
  step_keeps
    (fun s fields =>
       n_way_join_int_indexed_product import_type type_eqb n_way_join_rec s sh
         fields).
Proof.
  intros s s' fields answer Hstep.
  unfold n_way_join_int_indexed_product in Hstep.
  destruct fields as [ | [id1 first_fields] other_fields]; [ discriminate | ].
  cbv zeta in Hstep.
  match type of Hstep with
  | (if ?test then _ else _) = _ => destruct test
  end.
  - match type of Hstep with
    | context [find ?f ?l] => destruct (find f l) as [[id fields_of_id] | ]
    end; [ | discriminate ].
    destruct (import_types import_type s fields_of_id) as [s1 | ]
      eqn:Himported; [ | discriminate ].
    apply import_types_keeps in Himported. finish Hstep.
  - exact (join_fields_keeps _ _ _ _ _ _ _ Hstep).
Qed.

Lemma n_way_join_closures_entry_keeps :
  step_keeps (n_way_join_closures_entry n_way_join_rec).
Proof.
  intros s s' entries answer Hstep.
  unfold n_way_join_closures_entry,
    n_way_join_function_slot_indexed_product,
    n_way_join_value_slot_indexed_product in Hstep.
  cbv zeta in Hstep.
  match type of Hstep with
  | context [n_way_join_generic_product ?f s ?l] =>
    destruct (n_way_join_generic_product f s l) as [[closure_types s1] | ]
      eqn:Hclosures; [ | discriminate ]
  end.
  apply n_way_join_generic_product_keeps in Hclosures.
  match type of Hstep with
  | context [for_all_bindings_b ?p ?m] =>
    destruct (for_all_bindings_b p m); [ | discriminate ]
  end.
  match type of Hstep with
  | context [n_way_join_generic_product ?f s1 ?l] =>
    destruct (n_way_join_generic_product f s1 l) as [[value_slot_types s2] | ]
      eqn:Hvalue_slots; [ | discriminate ]
  end.
  apply n_way_join_generic_product_keeps in Hvalue_slots. finish Hstep.
Qed.

Lemma n_way_join_row_like_for_blocks_keeps :
  step_keeps
    (n_way_join_row_like_for_blocks import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' row_likes answer Hstep.
  unfold n_way_join_row_like_for_blocks in Hstep.
  set (split := List.fold_left _ row_likes Or_bottom.Bottom) in Hstep.
  cbv zeta in Hstep.
  destruct split as [[[known other] alloc_mode] | ]; [ | finish Hstep ].
  unfold map_join_result in Hstep.
  match type of Hstep with
  | context [n_way_join_row_like ?a ?b ?c ?d ?e ?f s known other] =>
    destruct (n_way_join_row_like a b c d e f s known other)
      as [[joined s1] | ] eqn:Hrow; [ | discriminate ]
  end.
  apply n_way_join_row_like_keeps in Hrow; [ finish Hstep | | ].
  - exact n_way_join_int_indexed_product_keeps.
  - intros join Hjoin.
    exact
      (generic_merge_map_known_keeps Z.eqb z_eqb_decides_equality _ join
         Hjoin).
Qed.

Lemma n_way_join_row_like_for_closures_keeps :
  step_keeps
    (n_way_join_row_like_for_closures join_env_n_way_join_env_extension
       n_way_join_rec).
Proof.
  intros s s' closures answer Hstep.
  unfold n_way_join_row_like_for_closures in Hstep. cbv zeta in Hstep.
  match type of Hstep with
  | context [n_way_join_row_like ?a ?b ?c ?d ?e ?f s ?k ?o] =>
    destruct (n_way_join_row_like a b c d e f s k o) as [[joined s1] | ]
      eqn:Hrow; [ | discriminate ]
  end.
  apply n_way_join_row_like_keeps in Hrow; [ finish Hstep | | ].
  - intros sh. exact n_way_join_closures_entry_keeps.
  - intros join Hjoin.
    apply (generic_merge_map_known_keeps Pos.eqb pos_eqb_decides_equality).
    intros s0 s0' cases b Hcall. cbv zeta in Hcall.
    match type of Hcall with
    | context [join s0 ?l] =>
      destruct (join s0 l) as [[[[c | ] s2] | ] | ] eqn:Hjoined;
        try discriminate
    end.
    apply Hjoin in Hjoined. finish Hcall.
Qed.

Lemma n_way_join_variant_keeps
    (s s' : state) (blocks : list (Index.t * Or_unknown.t row_like_for_blocks))
    (imms : list (Index.t * Or_unknown.t flambda_type))
    (extensions : list (Index.t * variant_extensions))
    (answer :
       Or_unknown.t
         (Or_unknown.t row_like_for_blocks * Or_unknown.t flambda_type
          * variant_extensions)) :
  n_way_join_variant import_type join_env_n_way_join_env_extension type_eqb
    block_shape_eqb n_way_join_rec s blocks imms extensions
  = Some (answer, s') ->
  keeps s s'.
Proof.
  intros Hstep. unfold n_way_join_variant in Hstep.
  match type of Hstep with
  | context [n_way_join_unknown ?f s blocks] =>
    destruct (n_way_join_unknown f s blocks) as [[blocks' sa] | ] eqn:Hblocks;
      [ | discriminate ]
  end.
  apply (n_way_join_unknown_keeps _ n_way_join_row_like_for_blocks_keeps)
    in Hblocks.
  destruct (n_way_join_unknown n_way_join_rec sa imms) as [[imms' sb] | ]
    eqn:Himms; [ | discriminate ].
  apply (n_way_join_unknown_keeps _ Hrec) in Himms.
  assert (Hsb : keeps s sb) by exact (keeps_trans _ _ _ Hblocks Himms).
  destruct (gather_extensions [] [] extensions)
    as [[when_immediates when_blocks] | ]; [ | finish Hstep ].
  cbv zeta in Hstep.
  destruct
    (n_way_join_env_extension join_env_n_way_join_env_extension sb
       when_immediates)
    as [[when_immediate sc] | ] eqn:Hwhen_immediate; [ | discriminate ].
  apply n_way_join_env_extension_keeps in Hwhen_immediate.
  destruct
    (n_way_join_env_extension join_env_n_way_join_env_extension sc
       when_blocks)
    as [[when_block sd] | ] eqn:Hwhen_block; [ | discriminate ].
  apply n_way_join_env_extension_keeps in Hwhen_block.
  assert (Hsd : keeps s sd).
  { exact
      (keeps_trans _ _ _ Hsb (keeps_trans _ _ _ Hwhen_immediate Hwhen_block)). }
  finish Hstep.
Qed.

Lemma n_way_join_head_of_kind_value_non_null_keeps :
  step_keeps
    (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' heads answer Hstep.
  unfold n_way_join_head_of_kind_value_non_null in Hstep.
  destruct heads as [ | [first_id first_head] other_heads];
    [ discriminate | ].
  destruct first_head
    as [is_int immediates get_tag blocks extensions is_unique | alloc_mode
       | k n alloc_mode | by_function_slot alloc_mode | strs].
  - cbv zeta in Hstep.
    destruct (gather_heads _ _ other_heads _) as [parts | ];
      [ | finish Hstep ].
    destruct (n_way_join_relation_simples n_way_join_simples s
                (parts_is_int parts))
      as [[is_int' sa] | ] eqn:His_int; [ | discriminate ].
    apply n_way_join_relation_simples_keeps in His_int.
    destruct (n_way_join_relation_simples n_way_join_simples sa
                (parts_get_tag parts))
      as [[get_tag' sb] | ] eqn:Hget_tag; [ | discriminate ].
    apply n_way_join_relation_simples_keeps in Hget_tag.
    unfold map_join_result in Hstep.
    match type of Hstep with
    | context [n_way_join_variant ?a ?b ?c ?d ?e sb ?bs0 ?is0 ?es0] =>
      destruct (n_way_join_variant a b c d e sb bs0 is0 es0)
        as [[joined sc] | ] eqn:Hvariant; [ | discriminate ]
    end.
    apply n_way_join_variant_keeps in Hvariant.
    assert (Hsc : keeps s sc).
    { exact (keeps_trans _ _ _ His_int (keeps_trans _ _ _ Hget_tag Hvariant)). }
    finish Hstep.
  - finish Hstep.
  - destruct (gather_heads _ _ other_heads _) as [[ns mode] | ];
      [ | finish Hstep ].
    unfold map_join_result in Hstep.
    destruct (n_way_join_rec s ns) as [[joined s1] | ] eqn:Hjoined;
      [ | discriminate ].
    apply Hrec in Hjoined. finish Hstep.
  - destruct (gather_heads _ _ other_heads _) as [[function_slots mode] | ];
      [ | finish Hstep ].
    destruct
      (n_way_join_row_like_for_closures join_env_n_way_join_env_extension
         n_way_join_rec s function_slots)
      as [[joined s1] | ] eqn:Hjoined; [ | discriminate ].
    apply n_way_join_row_like_for_closures_keeps in Hjoined. finish Hstep.
  - finish Hstep.
Qed.

Lemma n_way_join_head_of_kind_value_keeps :
  step_keeps
    (n_way_join_head_of_kind_value n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' heads answer Hstep.
  unfold n_way_join_head_of_kind_value in Hstep.
  destruct (n_way_join_is_null n_way_join_simples s heads)
    as [[is_null sa] | ] eqn:His_null; [ | discriminate ].
  apply n_way_join_is_null_keeps in His_null.
  destruct (non_null_heads [] heads) as [[ | first rest] | ];
    [ finish Hstep | | finish Hstep ].
  destruct
    (n_way_join_head_of_kind_value_non_null n_way_join_simples import_type
       join_env_n_way_join_env_extension type_eqb block_shape_eqb
       n_way_join_rec sa (first :: rest))
    as [[joined sb] | ] eqn:Hjoined; [ | discriminate ].
  apply n_way_join_head_of_kind_value_non_null_keeps in Hjoined.
  assert (Hsb : keeps s sb) by exact (keeps_trans _ _ _ His_null Hjoined).
  finish Hstep.
Qed.

Lemma n_way_join_expanded_head_keeps (kind : flambda_kind)
    (expandeds : list (Index.t * Expanded_type.t)) :
  ∀ (s s' : state) (answer : Or_unknown.t Expanded_type.t),
    n_way_join_expanded_head joined_env n_way_join_simples import_type
      join_env_n_way_join_env_extension reduce_head_of_kind_naked_immediate
      type_eqb block_shape_eqb n_way_join_rec s kind expandeds
    = Some (answer, s') ->
    keeps s s'.
Proof.
  induction expandeds as [ | [id1 expanded1] rest IH];
    intros s s' answer Hstep.
  { injection Hstep as <- <-. apply keeps_refl. }
  cbn [n_way_join_expanded_head] in Hstep.
  destruct (Expanded_type.descr expanded1) as [ | descr1 | ];
    [ finish Hstep | | exact (IH _ _ _ Hstep) ].
  destruct descr1 as [head1 | head1 | k head1 | head1 | head1];
    (destruct (extract_heads _ rest) as [[heads | ] | ];
     [ | finish Hstep | discriminate ]);
    unfold map_join_result in Hstep.
  - match type of Hstep with
    | context [n_way_join_head_of_kind_value ?a ?b ?c ?d ?e ?f s ?l] =>
      destruct (n_way_join_head_of_kind_value a b c d e f s l)
        as [[joined s1] | ] eqn:Hjoined; [ | discriminate ]
    end.
    apply n_way_join_head_of_kind_value_keeps in Hjoined. finish Hstep.
  - match type of Hstep with
    | context [n_way_join_head_of_kind_naked_immediate ?a ?b s ?l] =>
      destruct (n_way_join_head_of_kind_naked_immediate a b s l)
        as [[joined s1] | ] eqn:Hjoined; [ | discriminate ]
    end.
    apply n_way_join_head_of_kind_naked_immediate_keeps in Hjoined.
    finish Hstep.
  - unfold n_way_join_head_of_kind_naked_numbers,
      n_way_join_head_of_kind_naked_number in Hstep.
    finish Hstep.
  - unfold n_way_join_head_of_kind_rec_info in Hstep. finish Hstep.
  - unfold n_way_join_head_of_kind_region in Hstep. finish Hstep.
Qed.

Lemma n_way_join_body_keeps :
  step_keeps
    (n_way_join_body joined_env n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb
       n_way_join_rec).
Proof.
  intros s s' ts answer Hstep. unfold n_way_join_body in Hstep.
  destruct ts as [ | [id1 t1] others]; [ discriminate | ].
  cbv zeta in Hstep.
  match type of Hstep with
  | (if ?test then _ else _) = _ => destruct test; [ discriminate | ]
  end.
  destruct (alias_simples _) as [simples | ].
  - destruct (n_way_join_simples s (kind t1) simples) as [[joined s1] | ]
      eqn:Hjoined; [ | discriminate ].
    apply Hsimples in Hjoined. finish Hstep.
  - destruct (expand_heads _ _ s _) as [expandeds | ]; [ | discriminate ].
    unfold map_join_result in Hstep.
    match type of Hstep with
    | context [n_way_join_expanded_head ?a ?b ?c ?d ?e ?f ?g ?h s ?k ?l] =>
      destruct (n_way_join_expanded_head a b c d e f g h s k l)
        as [[joined s1] | ] eqn:Hjoined; [ | discriminate ]
    end.
    apply n_way_join_expanded_head_keeps in Hjoined. finish Hstep.
Qed.

End Body.

Lemma n_way_join_keeps_at (fuel : nat) :
  step_keeps
    (n_way_join joined_env n_way_join_simples import_type
       join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' ts answer Hstep. discriminate.
  - exact (n_way_join_body_keeps _ IH).
Qed.

End Keeps.

Theorem n_way_join_keeps : n_way_join_keeps_statement.
Proof.
  intros state keeps joined_env n_way_join_simples import_type
    join_env_n_way_join_env_extension expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel Hrefl
    Htrans Hsimples Himport Hextension s s' ts joined Hstep.
  exact
    (n_way_join_keeps_at keeps Hrefl Htrans joined_env n_way_join_simples
       import_type join_env_n_way_join_env_extension expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb Hsimples
       Himport Hextension fuel s s' ts joined Hstep).
Qed.

Theorem n_way_join_keeps_definitions
  : n_way_join_keeps_definitions_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb s s' ts joined Hstep.
  refine
    (n_way_join_keeps Join_env.t keeps_definitions Join_env.joined_env
       (Join_env.n_way_join_simples p fuel_of_env)
       (Join_env.import_type p fuel_of_env)
       (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
       keeps_definitions_refl keeps_definitions_trans _ _ _ s s' ts joined
       Hstep).
  - intros s0 s0' kind simples result Hsimples.
    exact
      (n_way_join_simples_keeps_definitions p fuel_of_env s0 s0' kind
         simples result Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_keeps_definitions p fuel_of_env s0 s0' ty Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact
      (n_way_join_env_extension_keeps_definitions p fuel_of_env s0 s0'
         extensions joined0 Hextension).
Qed.
