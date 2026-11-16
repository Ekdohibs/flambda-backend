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


(* The removal of variables from a type, under renaming: lemmas for
   FlambdaEnvRenameJoinProofs.v. Nothing here needs review.

   [project_variables_out] (FlambdaTypeProject.v) is given the
   variables to remove and the type to put for each. Given the renamed
   variables, and a function that puts for a renamed variable the
   renamed type, it answers for a renamed type the renamed answer. The
   same holds of the function defined with it for each structure of
   the grammar. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaTypeProject.
Import ListNotations.

Lemma traverse_renamed {A A' B B' : Type} (ga : A -> A') (gb : B -> B')
    (f : A -> option B) (f' : A' -> option B') (l : list A) :
  (∀ a : A, a ∈ l -> f' (ga a) = option_map gb (f a)) ->
  traverse f' (map ga l) = option_map (map gb) (traverse f l).
Proof.
  induction l as [ | a l IH]; intros Hf; [ reflexivity | ].
  cbn [map traverse].
  rewrite (Hf a (or_introl eq_refl)).
  destruct (f a) as [b | ]; cbn [option_map]; [ | reflexivity ].
  rewrite IH by (intros a' Hin; apply Hf; right; exact Hin).
  destruct (traverse f l); reflexivity.
Qed.

Section Projected.

Variable r : renaming.
Variable to_project : list variable.
Variables expand expand' : variable -> option flambda_type.

Hypothesis expand_renamed :
  ∀ var : variable,
    expand' (rename_variable r var) = option_map (rename_type r) (expand var).

Local Abbreviation renamed_to_project := (map (rename_variable r) to_project).

Lemma among_renamed_to_project (var : variable) :
  among (rename_variable r var) renamed_to_project = among var to_project.
Proof.
  unfold among.
  induction to_project as [ | var' vars IH]; cbn [map existsb];
    [ reflexivity | ].
  rewrite rename_variable_eqb, IH. reflexivity.
Qed.

Lemma project_relation_renamed (relation : option variable) :
  project_relation renamed_to_project expand'
    (option_map (rename_variable r) relation)
  = option_map (option_map (rename_variable r))
      (project_relation to_project expand relation).
Proof.
  destruct relation as [var | ]; [ | reflexivity ].
  unfold project_relation. cbn [option_map].
  rewrite among_renamed_to_project.
  destruct (among var to_project); cbn [negb]; [ | reflexivity ].
  rewrite expand_renamed.
  destruct (expand var) as [ty | ]; cbn [option_map]; [ | reflexivity ].
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt ty) as [[var' | sym | c] | ]; reflexivity.
Qed.

(* A relation and the name that it is of. *)
Local Abbreviation rename_relation :=
  (fun relation : Relation.t * name =>
     (fst relation, rename_name r (snd relation))).

Lemma project_inverse_relation_renamed (pair : Relation.t * name) :
  project_inverse_relation renamed_to_project expand' (rename_relation pair)
  = option_map (map rename_relation)
      (project_inverse_relation to_project expand pair).
Proof.
  destruct pair as [relation [var | sym]]; unfold project_inverse_relation;
    cbn [fst snd rename_name]; [ | reflexivity ].
  rewrite among_renamed_to_project.
  destruct (among var to_project); cbn [negb]; [ | reflexivity ].
  rewrite expand_renamed.
  destruct (expand var) as [ty | ]; cbn [option_map]; [ | reflexivity ].
  rewrite get_alias_opt_renamed.
  destruct (get_alias_opt ty) as [[var' | sym | c] | ]; reflexivity.
Qed.

Lemma project_head_of_kind_naked_immediate_renamed
    (head : head_of_kind_naked_immediate) :
  project_head_of_kind_naked_immediate renamed_to_project expand'
    (rename_head_of_kind_naked_immediate r head)
  = option_map (rename_head_of_kind_naked_immediate r)
      (project_head_of_kind_naked_immediate to_project expand head).
Proof.
  destruct head as [imms relations].
  unfold project_head_of_kind_naked_immediate.
  cbn [rename_head_of_kind_naked_immediate naked_immediates
       inverse_relations].
  rewrite
    (traverse_renamed rename_relation (map rename_relation)
       (project_inverse_relation to_project expand))
    by (intros pair _; apply project_inverse_relation_renamed).
  destruct (traverse (project_inverse_relation to_project expand) relations)
    as [projected | ];
    cbn [option_map rename_head_of_kind_naked_immediate naked_immediates
         inverse_relations];
    [ | reflexivity ].
  unfold rename_head_of_kind_naked_immediate.
  cbn [naked_immediates inverse_relations].
  rewrite List.concat_map. reflexivity.
Qed.

Lemma project_descr_renamed {head : Type} (rename_head : head -> head)
    (project_head project_head' : head -> option head)
    (of_expanded : flambda_type -> option (type_descr head))
    (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) ->
    project_head' (rename_head h) = option_map rename_head (project_head h))
  ->
  (∀ t : flambda_type,
    of_expanded (rename_type r t)
    = option_map (rename_descr r rename_head) (of_expanded t)) ->
  project_descr renamed_to_project expand' project_head' of_expanded
    (rename_descr r rename_head ty)
  = option_map (rename_descr r rename_head)
      (project_descr to_project expand project_head of_expanded ty).
Proof.
  intros Hhead Hof.
  destruct ty as [ | [h | [var | sym | c]] | ]; unfold project_descr;
    cbn [rename_descr or_unknown_or_bottom_map rename_simple];
    try reflexivity.
  - rewrite (Hhead h eq_refl). destruct (project_head h); reflexivity.
  - rewrite among_renamed_to_project.
    destruct (among var to_project); [ | reflexivity ].
    rewrite expand_renamed.
    destruct (expand var) as [expanded | ]; cbn [option_map];
      [ apply Hof | reflexivity ].
Qed.

Let type_projected (t : flambda_type) : Prop :=
  project_variables_out renamed_to_project expand' (rename_type r t)
  = option_map (rename_type r) (project_variables_out to_project expand t).

Let extension_projected (extension : env_extension) : Prop :=
  project_env_extension renamed_to_project expand'
    (rename_env_extension r extension)
  = option_map (rename_env_extension r)
      (project_env_extension to_project expand extension).

Lemma project_block_case_renamed (case : row_like_block_case) :
  P_block_case type_projected extension_projected case ->
  project_block_case
    (project_variables_out renamed_to_project expand')
    (project_env_extension renamed_to_project expand')
    (rename_block_case (rename_type r) (rename_env_extension r) case)
  = option_map (rename_block_case (rename_type r) (rename_env_extension r))
      (project_block_case (project_variables_out to_project expand)
         (project_env_extension to_project expand) case).
Proof.
  destruct case as [maps_to index extension].
  intros [Htypes Hextension].
  cbn [case_maps_to case_env_extension] in Htypes, Hextension.
  unfold project_block_case, rename_block_case.
  cbn [case_maps_to case_index case_env_extension].
  rewrite Hextension.
  destruct (project_env_extension to_project expand extension)
    as [extension' | ];
    cbn [option_map]; [ | reflexivity ].
  rewrite
    (traverse_renamed (rename_type r) (rename_type r)
       (project_variables_out to_project expand))
    by exact Htypes.
  destruct (traverse (project_variables_out to_project expand) maps_to);
    reflexivity.
Qed.

Lemma project_variables_out_renamed_all :
  (∀ t : flambda_type, type_projected t)
  ∧ (∀ head : head_of_kind_value,
      project_head_of_kind_value renamed_to_project expand'
        (rename_head_of_kind_value r head)
      = option_map (rename_head_of_kind_value r)
          (project_head_of_kind_value to_project expand head))
  ∧ (∀ head : head_of_kind_value_non_null,
      project_head_of_kind_value_non_null renamed_to_project expand'
        (rename_head_of_kind_value_non_null r head)
      = option_map (rename_head_of_kind_value_non_null r)
          (project_head_of_kind_value_non_null to_project expand head))
  ∧ (∀ blocks : row_like_for_blocks,
      project_row_like_for_blocks renamed_to_project expand'
        (rename_row_like_for_blocks r blocks)
      = option_map (rename_row_like_for_blocks r)
          (project_row_like_for_blocks to_project expand blocks))
  ∧ (∀ closures : row_like_for_closures,
      project_row_like_for_closures renamed_to_project expand'
        (rename_row_like_for_closures r closures)
      = option_map (rename_row_like_for_closures r)
          (project_row_like_for_closures to_project expand closures))
  ∧ (∀ entry : closures_entry,
      project_closures_entry renamed_to_project expand'
        (rename_closures_entry r entry)
      = option_map (rename_closures_entry r)
          (project_closures_entry to_project expand entry))
  ∧ (∀ extension : env_extension, extension_projected extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. unfold type_projected. simpl.
    rewrite
      (project_descr_renamed (rename_head_of_kind_value r)
         (project_head_of_kind_value to_project expand));
      [ | intros h ->; exact Hhead | intros [ | | | | ]; reflexivity ].
    destruct
      (project_descr to_project expand
         (project_head_of_kind_value to_project expand) value_of_expanded ty);
      reflexivity.
  - (* Type_naked_immediate *)
    intros ty. unfold type_projected. simpl.
    rewrite
      (project_descr_renamed (rename_head_of_kind_naked_immediate r)
         (project_head_of_kind_naked_immediate to_project expand));
      [ | intros h _; apply project_head_of_kind_naked_immediate_renamed
        | intros [ | | | | ]; reflexivity ].
    destruct
      (project_descr to_project expand
         (project_head_of_kind_naked_immediate to_project expand)
         naked_immediate_of_expanded ty);
      reflexivity.
  - (* Type_naked_number *)
    intros k ty. unfold type_projected. simpl.
    rewrite
      (project_descr_renamed (fun numbers : list Z => numbers)
         (fun head : list Z => Some head));
      [ | intros h _; reflexivity
        | intros [ty' | ty' | k' ty' | ty' | ty']; try reflexivity; simpl;
          destruct (kind_eqb (Naked_number k) (Naked_number k'));
          reflexivity ].
    destruct
      (project_descr to_project expand (fun head : list Z => Some head)
         (naked_number_of_expanded k) ty);
      reflexivity.
  - (* Type_rec_info *)
    intros ty. unfold type_projected. simpl.
    rewrite
      (project_descr_renamed (fun head : unit => head)
         (fun head : unit => Some head));
      [ | intros h _; reflexivity | intros [ | | | | ]; reflexivity ].
    destruct
      (project_descr to_project expand (fun head : unit => Some head)
         rec_info_of_expanded ty);
      reflexivity.
  - (* Type_region *)
    intros ty. unfold type_projected. simpl.
    rewrite
      (project_descr_renamed (fun head : unit => head)
         (fun head : unit => Some head));
      [ | intros h _; reflexivity | intros [ | | | | ]; reflexivity ].
    destruct
      (project_descr to_project expand (fun head : unit => Some head)
         region_of_expanded ty);
      reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct is_null as [ | is_null_var]; simpl;
      rewrite ?project_relation_renamed;
      try
        (destruct (project_relation to_project expand is_null_var)
           as [is_null_var' | ];
         cbn [option_map]; [ | reflexivity ]);
      (destruct non_null as [ | head | ]; simpl; try reflexivity);
      rewrite Hnon_null;
      (destruct (project_head_of_kind_value_non_null to_project expand head));
      reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    simpl. rewrite !project_relation_renamed.
    destruct (project_relation to_project expand is_int) as [is_int' | ];
      cbn [option_map]; [ | reflexivity ].
    destruct (project_relation to_project expand get_tag) as [get_tag' | ];
      cbn [option_map]; [ | reflexivity ].
    destruct immediates as [immediates | ]; cbn [or_unknown_map].
    all: try rewrite Himmediates.
    all:
      try
        (destruct (project_variables_out to_project expand immediates)
           as [immediates' | ];
         cbn [option_map]; [ | reflexivity ]).
    all: destruct blocks as [blocks | ]; cbn [or_unknown_map].
    all: try rewrite Hblocks.
    all:
      try
        (destruct (project_row_like_for_blocks to_project expand blocks)
           as [blocks' | ];
         cbn [option_map]; [ | reflexivity ]).
    all: destruct extensions as [ | when_immediate when_block];
      try reflexivity.
    all: destruct Hextensions as [Himmediate Hblock].
    all: unfold extension_projected in Himmediate, Hblock.
    all: rewrite Himmediate.
    all:
      destruct (project_env_extension to_project expand when_immediate)
        as [when_immediate' | ];
      cbn [option_map]; [ | reflexivity ].
    all: rewrite Hblock.
    all:
      destruct (project_env_extension to_project expand when_block);
      reflexivity.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. simpl.
    unfold type_projected in Hcontents. rewrite Hcontents.
    destruct (project_variables_out to_project expand contents);
      reflexivity.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. simpl. rewrite Hclosures.
    destruct
      (project_row_like_for_closures to_project expand by_function_slot);
      reflexivity.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    unfold Keyed.map_values.
    rewrite
      (traverse_renamed
         (fun binding : Z * Or_unknown.t row_like_block_case =>
            (fst binding,
             or_unknown_map
               (rename_block_case (rename_type r) (rename_env_extension r))
               (snd binding)))
         (fun binding : Z * Or_unknown.t row_like_block_case =>
            (fst binding,
             or_unknown_map
               (rename_block_case (rename_type r) (rename_env_extension r))
               (snd binding)))
         (fun binding : Z * Or_unknown.t row_like_block_case =>
            match snd binding with
            | Or_unknown.Known case =>
              let* case :=
                project_block_case (project_variables_out to_project expand)
                  (project_env_extension to_project expand) case
              in
              Some (fst binding, Or_unknown.Known case)
            | Or_unknown.Unknown => Some (fst binding, Or_unknown.Unknown)
            end)).
    + destruct
        (traverse
           (fun binding : Z * Or_unknown.t row_like_block_case =>
              match snd binding with
              | Or_unknown.Known case =>
                let* case :=
                  project_block_case
                    (project_variables_out to_project expand)
                    (project_env_extension to_project expand) case
                in
                Some (fst binding, Or_unknown.Known case)
              | Or_unknown.Unknown => Some (fst binding, Or_unknown.Unknown)
              end)
           known_tags)
        as [known_tags' | ];
        cbn [option_map]; [ | reflexivity ].
      destruct other_tags as [case | ]; cbn [or_bottom_map];
        [ | reflexivity ].
      rewrite (project_block_case_renamed case Hother).
      destruct
        (project_block_case (project_variables_out to_project expand)
           (project_env_extension to_project expand) case);
        reflexivity.
    + intros [tag [case | ]] Hin; cbn [fst snd or_unknown_map];
        [ | reflexivity ].
      rewrite (project_block_case_renamed case (Hknown tag case Hin)).
      destruct
        (project_block_case (project_variables_out to_project expand)
           (project_env_extension to_project expand) case);
        reflexivity.
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. unfold map_values.
    match goal with
    | |- context [@traverse ?A ?B ?f known_closures] =>
      rewrite
        (@traverse_renamed A A B _
           (fun binding =>
              (fst binding,
               rename_closures_case (rename_closures_entry r)
                 (rename_env_extension r) (snd binding)))
           (fun binding =>
              (fst binding,
               rename_closures_case (rename_closures_entry r)
                 (rename_env_extension r) (snd binding)))
           f);
      [ destruct (traverse f known_closures); reflexivity | ]
    end.
    intros [slot [entry index extension]] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    cbn [case_maps_to case_env_extension] in Hentry, Hextension.
    unfold extension_projected in Hextension.
    cbn [fst snd rename_closures_case case_maps_to case_index
         case_env_extension].
    rewrite Hextension.
    destruct (project_env_extension to_project expand extension)
      as [extension' | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite Hentry.
    destruct (project_closures_entry to_project expand entry); reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. unfold map_values.
    match goal with
    | |- context [@traverse ?A ?B ?f closure_types] =>
      rewrite
        (@traverse_renamed A A B _
           (fun binding =>
              (fst binding, rename_type r (snd binding)))
           (fun binding =>
              (fst binding, rename_type r (snd binding)))
           f);
      [ destruct (traverse f closure_types) as [closure_types' | ];
        cbn [option_map]; [ | reflexivity ]
      | intros [slot ty] Hin; cbn [fst snd];
        rewrite (Hclosures slot ty Hin);
        destruct (project_variables_out to_project expand ty);
        reflexivity ]
    end.
    match goal with
    | |- context [@traverse ?A ?B ?f value_slot_types] =>
      rewrite
        (@traverse_renamed A A B _
           (fun binding =>
              (fst binding, rename_type r (snd binding)))
           (fun binding =>
              (fst binding, rename_type r (snd binding)))
           f);
      [ destruct (traverse f value_slot_types); reflexivity
      | intros [slot ty] Hin; cbn [fst snd];
        rewrite (Hslots slot ty Hin);
        destruct (project_variables_out to_project expand ty);
        reflexivity ]
    end.
  - (* Env_extension *)
    intros equations Htypes. unfold extension_projected. simpl.
    match goal with
    | |- context [@traverse ?A ?B ?f equations] =>
      rewrite
        (@traverse_renamed A A B _
           (fun equation =>
              (rename_name r (fst equation), rename_type r (snd equation)))
           (map
              (fun equation =>
                 (rename_name r (fst equation),
                  rename_type r (snd equation))))
           f);
      [ destruct (traverse f equations) as [projected | ];
        cbn [option_map]; [ | reflexivity ]
      | ]
    end.
    + simpl. rewrite List.concat_map. reflexivity.
    + intros [[var | sym] ty] Hin; cbn [fst snd rename_name];
        rewrite ?among_renamed_to_project;
        try (destruct (among var to_project); [ reflexivity | ]);
        rewrite (Htypes _ ty Hin);
        destruct (project_variables_out to_project expand ty);
        reflexivity.
Qed.

End Projected.

(* The type from which variables are removed. *)
Lemma project_variables_out_renamed (r : renaming)
    (to_project : list variable)
    (expand expand' : variable -> option flambda_type) (t : flambda_type) :
  (∀ var : variable,
    expand' (rename_variable r var) = option_map (rename_type r) (expand var))
  ->
  project_variables_out (map (rename_variable r) to_project) expand'
    (rename_type r t)
  = option_map (rename_type r) (project_variables_out to_project expand t).
Proof.
  intros Hexpand.
  exact
    (proj1 (project_variables_out_renamed_all r to_project expand expand'
              Hexpand)
       t).
Qed.
