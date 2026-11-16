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


(* The meets of the parts of a type that are not values commute with a
   renaming of variables, and so do the meets of products. Lemmas only:
   nothing here is a statement to review.

   For the functions of FlambdaMeet.v from [meet_relation] to
   [meet_closures_entry], and from [reduce_inverse_relations] to
   [meet_head_of_kind_region], a lemma named after the function says
   that on the renamed environment and renamed arguments it gives the
   renamed answer; where the function is a meet of two parts, the
   lemma is [meet_of_parts_related] of it (FlambdaMeetRenameSpec.v).
   The functions are given a meet of less fuel: there are two, one
   with the arguments and one with the renamed arguments, which the
   renaming relates ([meet_expanded_head_related]); a hypothesis of
   the section. A renaming of variables fixes the type that the
   choices give a name of which a number is the tag
   ([meet_choices_closed]): a second hypothesis, shown below of the
   two records of choices that FlambdaMeet.v has. The environment of
   the meet is FlambdaEnvRenameMeetProofs.v, and the functions that
   put a meet together FlambdaEnvRenameMeetPartsProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaTypingEnv FlambdaEnvRenameSpec
  FlambdaEnvRenameProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaProversRenameSpec FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetRenameSpec FlambdaEnvRenameMeetProofs
  FlambdaEnvRenameMeetPartsProofs.
Import ListNotations.

(* ======================================================================== *)
(* Parts that hold no name                                                  *)
(* ======================================================================== *)

(* The functions of FlambdaEnvRenameMeetPartsProofs.v, for parts that a
   renaming leaves as they are. *)

Lemma map_values_fixed {K A : Type} (bindings : Keyed.finite_map K A) :
  Keyed.map_values (fun a => a) bindings = bindings.
Proof.
  unfold Keyed.map_values.
  induction bindings as [ | [key a] bindings IH]; cbn [map fst snd];
    [ | rewrite IH ]; reflexivity.
Qed.

Lemma map_meet_of_fixed_renamed {K A : Type} (r : renaming)
    (eqb : K -> K -> bool)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (env : Meet_env.t) (left right : Keyed.finite_map K A) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_data' (rename_meet_env r env) a1 a2
    = option_map (rename_meet_result r (fun a => a)) (meet_data env a1 a2)) ->
  map_meet eqb meet_data' (rename_meet_env r env) left right
  = option_map (rename_meet_result r (fun m => m))
      (map_meet eqb meet_data env left right).
Proof.
  intros Hmeet. unfold map_meet. cbv zeta.
  apply (meet_mapping_renamed r (fun a => a) (fun m => m));
    [ exact Hmeet | | ].
  - intros step step' init Hstep.
    assert (Hstep' :
      ∀ (key : K) (data_left data_right : option A)
        (state : mapping_state K A),
        step' key data_left data_right
          (rename_mapping_state r (fun a => a) state)
        = rename_mapping_state r (fun a => a)
            (step key data_left data_right state)).
    { intros key data_left data_right state.
      specialize (Hstep key data_left data_right state).
      destruct data_left, data_right; exact Hstep. }
    unfold Keyed.fold2_on. revert init.
    induction (Keyed.keys_of_both eqb left right) as [ | key keys IH];
      intros init; cbn [List.fold_left]; [ reflexivity | ].
    rewrite Hstep'. apply IH.
  - intros l. rewrite map_values_fixed. reflexivity.
Qed.

Lemma meet_unknown_of_fixed_renamed {A : Type} (r : renaming)
    (meet_contents meet_contents' :
      Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) (env : Meet_env.t)
    (or_unknown1 or_unknown2 : Or_unknown.t A) :
  (∀ (env : Meet_env.t) (a1 a2 : A),
    meet_contents' (rename_meet_env r env) a1 a2
    = option_map (rename_meet_result r (fun a => a))
        (meet_contents env a1 a2)) ->
  meet_unknown meet_contents' contents_is_bottom (rename_meet_env r env)
    or_unknown1 or_unknown2
  = option_map (rename_meet_result r (fun or_unknown => or_unknown))
      (meet_unknown meet_contents contents_is_bottom env or_unknown1
         or_unknown2).
Proof.
  intros Hmeet.
  destruct or_unknown1 as [contents1 | ], or_unknown2 as [contents2 | ];
    cbn [meet_unknown].
  - destruct (contents_is_bottom contents1), (contents_is_bottom contents2);
      cbn [andb]; try reflexivity.
    rewrite Hmeet.
    destruct (meet_contents env contents1 contents2) as [result | ];
      cbn [option_map]; [ | reflexivity ].
    f_equal. apply map_result_renamed. intros a. reflexivity.
  - destruct (contents_is_bottom contents1); reflexivity.
  - destruct (contents_is_bottom contents2); reflexivity.
  - reflexivity.
Qed.

(* A function type holds no name, and the meet of two answers no new
   one: whatever [g] is. *)
Lemma meet_function_type_renamed (r : renaming)
    (g : Or_unknown.t function_type -> Or_unknown.t function_type)
    (env : Meet_env.t) (func_type1 func_type2 : Or_unknown.t function_type) :
  meet_function_type (rename_meet_env r env) func_type1 func_type2
  = option_map (rename_meet_result r g)
      (meet_function_type env func_type1 func_type2).
Proof.
  destruct func_type1 as [func_type1 | ], func_type2 as [func_type2 | ];
    cbn [meet_function_type option_map]; try reflexivity.
  unfold meet_code_id.
  destruct
    (Pos.eqb (function_type_code_id func_type1)
       (function_type_code_id func_type2));
    reflexivity.
Qed.

(* ======================================================================== *)
(* The meets that are given the meet of less fuel                           *)
(* ======================================================================== *)

Section Meet_given.

(* One renaming and two meets of expanded types of less fuel, which the
   renaming relates. *)
Variable r : renaming.
Variables m m' : meet_expanded_head.
Hypothesis m_related : meet_expanded_head_related r m m'.

(* [meet], of FlambdaMeet.v: the meet of two types. *)
Lemma meet_of_types_renamed :
  meet_of_parts_related r (rename_type r) (rename_type r) (meet m) (meet m').
Proof. intros env t1 t2. apply (meet_type_renamed r m m' m_related). Qed.

Lemma meet_relation_renamed :
  meet_of_parts_related r (option_map (rename_variable r))
    (option_map (rename_variable r)) (meet_relation m) (meet_relation m').
Proof.
  intros env [var1 | ] [var2 | ]; cbn [meet_relation option_map];
    try reflexivity.
  change (Var (rename_variable r var1)) with (rename_simple r (Var var1)).
  change (Var (rename_variable r var2)) with (rename_simple r (Var var2)).
  rewrite (add_alias_renamed r m m' m_related).
  destruct (Meet_env.add_alias env (Var var1) (Var var2) m)
    as [[env' | ] | ];
    reflexivity.
Qed.

Lemma meet_array_of_types_renamed (env : Meet_env.t)
    (fields1 fields2 : list flambda_type) (length : nat) :
  meet_array_of_types m' (rename_meet_env r env) (map (rename_type r) fields1)
    (map (rename_type r) fields2) length
  = option_map (rename_meet_result r (map (rename_type r)))
      (meet_array_of_types m env fields1 fields2 length).
Proof.
  unfold meet_array_of_types. cbv zeta.
  apply (meet_mapping_renamed r (rename_type r) (map (rename_type r)));
    [ exact meet_of_types_renamed | | ].
  - intros step step' init Hstep. revert init.
    induction (seq 0 length) as [ | i indices IH]; intros init;
      cbn [List.fold_left]; [ reflexivity | ].
    rewrite !nth_error_map, Hstep. apply IH.
  - intros [ | [key data] l]; [ reflexivity | ].
    cbn [Keyed.map_values map fst snd]. rewrite map_map.
    apply map_ext. intros i.
    change ((key, rename_type r data)
            :: map
                 (fun binding : nat * flambda_type =>
                    (fst binding, rename_type r (snd binding)))
                 l)
      with (Keyed.map_values (rename_type r) ((key, data) :: l)).
    rewrite keyed_lookup_map_values.
    destruct (Keyed.lookup Nat.eqb i ((key, data) :: l)); reflexivity.
Qed.

Lemma meet_int_indexed_product_renamed :
  meet_of_parts_related r (map (rename_type r)) (map (rename_type r))
    (meet_int_indexed_product m) (meet_int_indexed_product m').
Proof.
  intros env fields1 fields2. unfold meet_int_indexed_product. cbv zeta.
  rewrite !length_map. apply meet_array_of_types_renamed.
Qed.

Lemma meet_product_function_slot_indexed_renamed :
  meet_of_parts_related r (map_values (rename_type r))
    (map_values (rename_type r)) (meet_product_function_slot_indexed m)
    (meet_product_function_slot_indexed m').
Proof.
  intros env components1 components2.
  exact
    (map_meet_renamed r Pos.eqb (rename_type r) (meet m) (meet m') env
       components1 components2 meet_of_types_renamed).
Qed.

Lemma meet_product_value_slot_indexed_renamed :
  meet_of_parts_related r (map_values (rename_type r))
    (map_values (rename_type r)) (meet_product_value_slot_indexed m)
    (meet_product_value_slot_indexed m').
Proof.
  intros env components1 components2.
  exact
    (map_meet_renamed r Pos.eqb (rename_type r) (meet m) (meet m') env
       components1 components2 meet_of_types_renamed).
Qed.

Lemma meet_closures_entry_renamed :
  meet_of_parts_related r (rename_closures_entry r) (rename_closures_entry r)
    (meet_closures_entry m) (meet_closures_entry m').
Proof.
  intros env [function_types1 closure_types1 value_slot_types1]
    [function_types2 closure_types2 value_slot_types2].
  cbn [rename_closures_entry meet_closures_entry].
  apply
    (combine_results3_renamed r (fun function_types => function_types)
       (map_values (rename_type r)) (map_values (rename_type r))
       (rename_closures_entry r)).
  - intros env' function_types function_types'.
    apply map_meet_of_fixed_renamed. intros env'' func_type func_type'.
    apply meet_function_type_renamed.
  - exact meet_product_function_slot_indexed_renamed.
  - exact meet_product_value_slot_indexed_renamed.
  - reflexivity.
Qed.

(* ======================================================================== *)
(* Naked immediates, naked numbers, and the kinds with one type             *)
(* ======================================================================== *)

(* Where the meet can do other than the compiler. A renaming of
   variables fixes the type that the choices give a name of which a
   number among some tags is [Get_tag]. *)
Variable choices : meet_choices.
Hypothesis choices_closed : meet_choices_closed choices.

(* An equation whose type holds no name. *)
Lemma add_equation_strict_of_closed_renamed (env : Meet_env.t)
    (name : name) (ty : flambda_type) :
  rename_type r ty = ty ->
  Meet_env.add_equation_strict (rename_meet_env r env) (rename_name r name)
    ty m'
  = option_map (or_bottom_map (rename_meet_env r))
      (Meet_env.add_equation_strict env name ty m).
Proof.
  intros Hty. rewrite <- Hty at 1.
  apply (add_equation_strict_renamed r m m' m_related).
Qed.

Lemma reduce_inverse_relations_renamed (env : Meet_env.t)
    (imms : Or_unknown.t (list Z)) (relations : list (Relation.t * name)) :
  reduce_inverse_relations choices m' (rename_meet_env r env) imms
    (map
       (fun relation : Relation.t * name =>
          (fst relation, rename_name r (snd relation)))
       relations)
  = option_map (or_bottom_map (rename_meet_env r))
      (reduce_inverse_relations choices m env imms relations).
Proof.
  destruct imms as [imms | ]; cbn [reduce_inverse_relations];
    [ | reflexivity ].
  cbv zeta.
  match goal with
  | |- List.fold_left ?step' _ ?init'
       = option_map _ (List.fold_left ?step _ ?init) =>
    change init'
      with (option_map (or_bottom_map (rename_meet_env r)) init);
    apply
      (fold_left_renamed (option_map (or_bottom_map (rename_meet_env r)))
         (fun relation : Relation.t * name =>
            (fst relation, rename_name r (snd relation)))
         step step')
  end.
  intros [[env' | ] | ] [relation name];
    cbn [option_map or_bottom_map fst snd]; try reflexivity.
  destruct relation.
  - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
      try reflexivity;
      apply add_equation_strict_of_closed_renamed; reflexivity.
  - destruct (Keyed.among Z.eqb 0%Z imms), (Keyed.among Z.eqb 1%Z imms);
      try reflexivity;
      apply add_equation_strict_of_closed_renamed; reflexivity.
  - unfold type_of_get_tag_scrutinee_or_unknown.
    destruct
      (forallb is_structured_block (Keyed.dedup Z.eqb (filter is_tag imms)));
      [ | reflexivity ].
    apply add_equation_strict_of_closed_renamed, choices_closed.
Qed.

Lemma meet_head_of_kind_naked_immediate_renamed :
  meet_of_parts_related r (rename_head_of_kind_naked_immediate r)
    (rename_head_of_kind_naked_immediate r)
    (meet_head_of_kind_naked_immediate choices m)
    (meet_head_of_kind_naked_immediate choices m').
Proof.
  intros env t1 t2. unfold meet_head_of_kind_naked_immediate.
  rewrite current_typing_env_renamed,
    !reduce_head_of_kind_naked_immediate_renamed.
  destruct
    (reduce_head_of_kind_naked_immediate (Meet_env.current_typing_env env) t1)
    as [head1 | ],
    (reduce_head_of_kind_naked_immediate (Meet_env.current_typing_env env) t2)
    as [head2 | ];
    cbn [or_bottom_map option_map]; try reflexivity.
  change (naked_immediates (rename_head_of_kind_naked_immediate r head1))
    with (naked_immediates head1).
  change (naked_immediates (rename_head_of_kind_naked_immediate r head2))
    with (naked_immediates head2).
  change (inverse_relations (rename_head_of_kind_naked_immediate r head1))
    with (map
            (fun relation : Relation.t * name =>
               (fst relation, rename_name r (snd relation)))
            (inverse_relations head1)).
  change (inverse_relations (rename_head_of_kind_naked_immediate r head2))
    with (map
            (fun relation : Relation.t * name =>
               (fst relation, rename_name r (snd relation)))
            (inverse_relations head2)).
  rewrite
    (meet_unknown_of_fixed_renamed r
       (fun env s1 s2 => Some (set_meet Z.eqb (fun s => s) env s1 s2))
       (fun env s1 s2 => Some (set_meet Z.eqb (fun s => s) env s1 s2)));
    [ | intros env' s1 s2; cbn [option_map]; f_equal;
        apply set_meet_renamed; reflexivity ].
  destruct
    (meet_unknown
       (fun env s1 s2 => Some (set_meet Z.eqb (fun s => s) env s1 s2))
       Keyed.is_empty env (naked_immediates head1) (naked_immediates head2))
    as [[res | res env'] | ];
    cbn [option_map rename_meet_result]; try reflexivity.
  cbv zeta. rewrite meet_inverse_relations_renamed.
  cbn [left_inputs_only meet_return_value_of right_inputs_only].
  destruct res as [ | | | imms],
    (meet_return_value_of
       (meet_inverse_relations (inverse_relations head1)
          (inverse_relations head2)))
    as [ | | | relations];
    cbn [rename_return_value combine_meet_return_values extract_value
         sequence_return_value option_map rename_meet_result];
    rewrite ?head_of_kind_naked_immediate_from_descr_non_empty_renamed;
    try
      match goal with
      | |- context
             [head_of_kind_naked_immediate_from_descr_non_empty ?imms ?rels]
        =>
        destruct
          (head_of_kind_naked_immediate_from_descr_non_empty imms rels)
          as [head | ];
        cbn [option_map sequence_return_value]; [ | reflexivity ]
      end;
    rewrite ?reduce_inverse_relations_renamed;
    try
      match goal with
      | |- context [reduce_inverse_relations choices m env' ?imms ?rels] =>
        destruct (reduce_inverse_relations choices m env' imms rels)
          as [[env'' | ] | ]
      end;
    reflexivity.
Qed.

End Meet_given.

(* A set of naked numbers is not renamed. *)
Lemma meet_head_of_kind_naked_number_renamed (r : renaming)
    (g : list Z -> list Z) (env : Meet_env.t) (t1 t2 : list Z) :
  (∀ s : list Z, g s = s) ->
  meet_head_of_kind_naked_number (rename_meet_env r env) t1 t2
  = rename_meet_result r g (meet_head_of_kind_naked_number env t1 t2).
Proof. intros Hg. apply set_meet_renamed. exact Hg. Qed.

Lemma meet_head_of_kind_rec_info_renamed (r : renaming) (g : unit -> unit)
    (env : Meet_env.t) (t1 t2 : unit) :
  meet_head_of_kind_rec_info (rename_meet_env r env) t1 t2
  = rename_meet_result r g (meet_head_of_kind_rec_info env t1 t2).
Proof. reflexivity. Qed.

Lemma meet_head_of_kind_region_renamed (r : renaming) (g : unit -> unit)
    (env : Meet_env.t) (t1 t2 : unit) :
  meet_head_of_kind_region (rename_meet_env r env) t1 t2
  = rename_meet_result r g (meet_head_of_kind_region env t1 t2).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* The two records of choices                                               *)
(* ======================================================================== *)

(* A renaming of variables fixes the type that each gives a name of
   which a number among some tags is [Get_tag]. *)

Lemma compiler_choices_closed (r : renaming) (tags : list Z)
    (alloc_mode : Alloc_mode.t) :
  rename_type r (type_of_get_tag_scrutinee compiler_choices tags alloc_mode)
  = type_of_get_tag_scrutinee compiler_choices tags alloc_mode.
Proof. apply rename_type_blocks_with_these_tags. Qed.

Lemma repaired_choices_closed (r : renaming) (tags : list Z)
    (alloc_mode : Alloc_mode.t) :
  rename_type r (type_of_get_tag_scrutinee repaired_choices tags alloc_mode)
  = type_of_get_tag_scrutinee repaired_choices tags alloc_mode.
Proof.
  cbn [type_of_get_tag_scrutinee repaired_choices].
  unfold immediate_or_blocks_with_these_tags. simpl.
  unfold create_variant, non_null_value.
  do 7 f_equal. unfold Keyed.map_values. rewrite map_map. reflexivity.
Qed.
