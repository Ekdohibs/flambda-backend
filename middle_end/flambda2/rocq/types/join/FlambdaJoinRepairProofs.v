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


(* The proofs of the statements of FlambdaJoinRepairSpec.v. This file
   needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeMeaningBoolSpec
  FlambdaTypeMeaningBoolProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinAddsSpec FlambdaJoinTypes
  FlambdaJoinTypesWithEnvSpec FlambdaJoinTypesWithEnvProofs
  FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetExamplesSpec FlambdaJoinMeet
  FlambdaJoinMeetExamplesSpec FlambdaJoinMeetSpec
  FlambdaJoinMeetProofs FlambdaJoinMeetFitsSpec FlambdaJoinMeetFitsProofs
  FlambdaSimplifyPruningSpec FlambdaSimplifyPruningProofs
  FlambdaJoinRepairSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the repair gains                                                    *)
(* ======================================================================== *)

Theorem repaired_scrutinees_have_values
  : repaired_scrutinees_have_values_statement.
Proof.
  intros depth p fuel jt s1 s2 s3 equations existentials relations2
    relations3 index use statics valuation Hjt Hrounds Hloop Huse Hsatisfied
    Hbound r relation x v Hrecorded _.
  destruct (delayed_equations_record_where_chosen _ _ _ _ _ _ _ _ Hjt Hrounds)
    as [[Henvs1 _] _].
  destruct (loop_records_where_chosen _ _ _ _ _ _ _ _ _ _ Hjt Hloop)
    as [[Henvs2 _] _].
  assert (Huse1 : Join_env.joined_env s1 index = Some use).
  { rewrite <- (Henvs1 index), <- (Henvs2 index). exact Huse. }
  destruct (repaired_rounds_scrutinees_have_values depth p fuel jt s1 s2 s3
              equations existentials relations2 relations3 r relation
              (Name_var x) index use (fun _ => False) statics valuation Hjt
              Hrounds Hloop Hrecorded Huse1 Hsatisfied Hbound)
    as [x' [Heq Hvalue]].
  injection Heq as <-. rewrite lookup_target_valuation.
  destruct Hvalue
    as [[Hknown [w Hw]]
       | [simples [kind [simple [w [Hdefinition [Hfind Hw]]]]]]].
  - destruct (value_of_definition s3 index valuation x) as [w' | ];
      [ exists w'; reflexivity | ].
    unfold known_to. unfold knows in Hknown. rewrite Hknown.
    exists w. exact Hw.
  - unfold value_of_definition. rewrite Hdefinition, Hfind, Hw.
    exists w. reflexivity.
Qed.

Lemma model_join_of_types_keeps_definitions (join_depth : nat) (fuel : fuels) :
  join_of_types_keeps_definitions (model_join_of_types join_depth fuel).
Proof.
  unfold model_join_of_types.
  apply n_way_join_of_types_keeps_definitions.
Qed.

Theorem repaired_join_scoped_envs_satisfied
  : repaired_join_scoped_envs_satisfied_statement.
Proof.
  intros join_depth fuel meet_head initial_env result scoped_envs envs s1 s2
    s3 relations2 relations3 kept equations relation_equations definitions
    index scoped_env statics valuation join_choices initial_tenv p jt avoid
    s0 menv Hsound Hkeeps Hfits Hscoped Hagree Hjoin Hcut Haliases Hdelayed
    Hloop Hadds Huse Hstatics Hsat_initial Hsat_use Hbound Htags Haliases_v
    Hvalued.
  apply (model_join_scoped_envs_satisfied_outright join_choices join_depth
           fuel meet_head initial_env result scoped_envs envs s1 s2 s3
           relations2 relations3 kept equations relation_equations definitions
           index scoped_env statics valuation Hsound Hkeeps Hfits Hscoped
           Hagree Hjoin Hcut Haliases Hdelayed Hloop Hadds Huse Hstatics
           Hsat_initial Hsat_use Htags Haliases_v);
    [ | exact Hvalued ].
  exact (repaired_scrutinees_have_values join_depth p (fuel_of_env fuel) jt
           s1 s2 s3 _ _ relations2 relations3 index scoped_env statics
           valuation (model_join_of_types_keeps_definitions join_depth fuel)
           Hdelayed Hloop Huse Hsat_use Hbound).
Qed.

Theorem repaired_join_with_admits_use
  : repaired_join_with_admits_use_statement.
Proof.
  intros A join_depth fuel mh avoid inputs target analysis envs s3 relations3
    kept equations relation_equations definitions index use statics valuation
    join_choices Hmh Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin Hrun
    Hat Hbottom Hstatics Hsat_use Hsat_source Hbound Htags Haliases_v Hvalued.
  apply (model_join_with_admits_use_outright A join_choices join_depth fuel
           mh avoid inputs target analysis envs s3 relations3 kept equations
           relation_equations definitions index use statics valuation Hmh
           Hsound Hinputs Hkinded Hagree Hfits Hfit_uses Hjoin Hrun Hat
           Hbottom Hstatics Hsat_use Hsat_source Htags Haliases_v);
    [ | exact Hvalued ].
  pose proof Hrun
    as [s1 [s2 [relations2 [Hcut [Haliases [Hdelayed [Hloop Hadds]]]]]]].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (inputs_first_phases join_choices join_depth fuel avoid inputs envs
              s1 s2 s3 relations2 relations3 Hinputs Hkinded Hagree Hcut
              Haliases Hdelayed Hloop)
    as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Huse3 & _).
  assert (Huse : Join_env.joined_env s3 index = Some use).
  { apply Huse3. split; [ exact Hat | exact Hbottom ]. }
  exact (repaired_scrutinees_have_values join_depth
           (join_parameters join_depth) (fuel_of_env fuel)
           (model_join_of_types join_depth fuel) s1 s2 s3 _ _ relations2
           relations3 index use statics valuation
           (model_join_of_types_keeps_definitions join_depth fuel) Hdelayed
           Hloop Huse Hsat_use Hbound).
Qed.

(* ======================================================================== *)
(* The premise on the variables that may occur in terms is needed           *)
(* ======================================================================== *)

Definition in_terms_source_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types In_terms.source_env.

Definition in_terms_use_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types In_terms.use_env.

Definition in_terms_use_null_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types In_terms.use_env_null.

Lemma in_terms_source_names_to_types :
  Typing_env.names_to_types In_terms.source_env = in_terms_source_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma in_terms_use_names_to_types :
  Typing_env.names_to_types In_terms.use_env = in_terms_use_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma in_terms_use_null_names_to_types :
  Typing_env.names_to_types In_terms.use_env_null
  = in_terms_use_null_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma in_terms_use_at (index : Index.t) (use : Typing_env.t) :
  use_at In_terms.inputs index use ->
  use = In_terms.use_env ∨ use = In_terms.use_env_null.
Proof.
  intros [id Hnth]. cbn in Hnth.
  destruct index as [ | [ | index]]; cbn in Hnth.
  - injection Hnth as _ <-. left. reflexivity.
  - injection Hnth as _ <-. right. reflexivity.
  - destruct index; discriminate.
Qed.

Lemma in_terms_source_invariant : typing_env_invariant In_terms.source_env.
Proof.
  apply add_variable_definition_invariant; [ | reflexivity ].
  apply add_variable_definition_invariant; [ | reflexivity ].
  exact create_invariant.
Qed.

Lemma in_terms_use_invariant : typing_env_invariant In_terms.use_env.
Proof.
  apply replace_equation_invariant.
  - apply increment_scope_invariant. exact in_terms_source_invariant.
  - reflexivity.
  - intros y [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. discriminate.
  - left. reflexivity.
Qed.

Lemma in_terms_use_null_invariant :
  typing_env_invariant In_terms.use_env_null.
Proof.
  apply replace_equation_invariant.
  - apply increment_scope_invariant. exact in_terms_source_invariant.
  - reflexivity.
  - intros y [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. discriminate.
  - left. reflexivity.
Qed.

(* The cases of a lookup in the entries of an environment of this
   instance: the variable is [x], [r], or neither. *)
Ltac in_terms_cases Hlookup n :=
  destruct n as [y | sym]; cbn [Keyed.lookup name_eqb fst snd] in Hlookup;
  [ destruct (Pos.eqb y 2%positive) eqn:Hx;
    [ apply Pos.eqb_eq in Hx; subst y
    | destruct (Pos.eqb y 1%positive) eqn:Hr;
      [ apply Pos.eqb_eq in Hr; subst y | ] ]
  | ].

Lemma in_terms_source_no_bottom_types : no_bottom_types In_terms.source_env.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite in_terms_source_names_to_types in Hlookup.
  unfold in_terms_source_entries in Hlookup.
  in_terms_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma in_terms_use_no_bottom_types : no_bottom_types In_terms.use_env.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite in_terms_use_names_to_types in Hlookup.
  unfold in_terms_use_entries in Hlookup.
  in_terms_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma in_terms_use_null_no_bottom_types :
  no_bottom_types In_terms.use_env_null.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite in_terms_use_null_names_to_types in Hlookup.
  unfold in_terms_use_null_entries in Hlookup.
  in_terms_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma in_terms_source_kinded : typing_env_kinded In_terms.source_env.
Proof.
  apply add_variable_definition_kinded; [ reflexivity | reflexivity | ].
  apply add_variable_definition_kinded; [ reflexivity | reflexivity | ].
  exact create_kinded.
Qed.

Lemma in_terms_use_kinded : typing_env_kinded In_terms.use_env.
Proof.
  apply replace_equation_kinded.
  - eexists. vm_compute. reflexivity.
  - intros simple kind [Heq | []]. injection Heq as <- <-.
    eexists. vm_compute. reflexivity.
  - apply increment_scope_kinded. exact in_terms_source_kinded.
Qed.

Lemma in_terms_use_null_kinded : typing_env_kinded In_terms.use_env_null.
Proof.
  apply replace_equation_kinded.
  - eexists. vm_compute. reflexivity.
  - intros simple kind [Heq | []]. injection Heq as <- <-.
    eexists. vm_compute. reflexivity.
  - apply increment_scope_kinded. exact in_terms_source_kinded.
Qed.

Theorem in_terms_inputs_well_formed : in_terms_inputs_well_formed_statement.
Proof.
  split; [ split; [ | split; [ | split; [ | split ] ] ] | ].
  - split; [ exact in_terms_source_invariant | ].
    intros index use Huse.
    destruct (in_terms_use_at _ _ Huse); subst use;
      [ exact in_terms_use_invariant | exact in_terms_use_null_invariant ].
  - split; [ exact in_terms_source_no_bottom_types | ].
    intros index use Huse.
    destruct (in_terms_use_at _ _ Huse); subst use;
      [ exact in_terms_use_no_bottom_types
      | exact in_terms_use_null_no_bottom_types ].
  - intros index use Huse _.
    destruct (in_terms_use_at _ _ Huse); subst use;
      eexists; vm_compute; reflexivity.
  - intros index use sym Huse Hdefined.
    destruct (in_terms_use_at _ _ Huse); subst use; discriminate.
  - split.
    + intros y Hknown. unfold knows, Typing_env.mem in Hknown.
      change (source In_terms.inputs) with In_terms.source_env in Hknown.
      rewrite in_terms_source_names_to_types in Hknown.
      unfold in_terms_source_entries in Hknown.
      cbn [Keyed.lookup name_eqb fst snd] in Hknown.
      destruct (Pos.eqb y 2%positive) eqn:Hx;
        [ apply Pos.eqb_eq in Hx; subst y; reflexivity | ].
      destruct (Pos.eqb y 1%positive) eqn:Hr;
        [ apply Pos.eqb_eq in Hr; subst y; reflexivity | discriminate ].
    + intros index use y Huse Hknown.
      unfold knows, Typing_env.mem in Hknown.
      destruct (in_terms_use_at _ _ Huse); subst use;
        [ rewrite in_terms_use_names_to_types in Hknown;
          unfold in_terms_use_entries in Hknown
        | rewrite in_terms_use_null_names_to_types in Hknown;
          unfold in_terms_use_null_entries in Hknown ];
        cbn [Keyed.lookup name_eqb fst snd] in Hknown;
        (destruct (Pos.eqb y 2%positive) eqn:Hx;
           [ apply Pos.eqb_eq in Hx; subst y; reflexivity | ]);
        (destruct (Pos.eqb y 1%positive) eqn:Hr;
           [ apply Pos.eqb_eq in Hr; subst y; reflexivity
           | discriminate ]).
  - intros index use Huse.
    destruct (in_terms_use_at _ _ Huse); subst use;
      [ exact in_terms_use_kinded | exact in_terms_use_null_kinded ].
Qed.

Lemma in_terms_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics Witness.valuation In_terms.use_env
  ∧ typing_env_satisfied_strictly statics Witness.valuation
      In_terms.source_env.
Proof.
  split.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite in_terms_use_names_to_types in Hlookup.
      unfold in_terms_use_entries in Hlookup.
      in_terms_cases Hlookup n; try discriminate.
      injection Hlookup as <- _. injection Hvalue as <-.
      right. rewrite type_satisfied_naked_immediate. reflexivity.
    + change (aliases_respected Aliases.empty Witness.valuation).
      apply empty_respected.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite in_terms_source_names_to_types in Hlookup.
      unfold in_terms_source_entries in Hlookup.
      in_terms_cases Hlookup n; try discriminate.
      injection Hlookup as <- _. injection Hvalue as <-.
      right. rewrite type_satisfied_naked_immediate. reflexivity.
    + change (aliases_respected Aliases.empty Witness.valuation).
      apply empty_respected.
Qed.

Theorem in_terms_run : in_terms_run_statement.
Proof.
  split; [ exact in_terms_satisfied | ].
  intros Hbound. destruct (Hbound Witness.x eq_refl) as [v Hv].
  discriminate Hv.
Qed.

Definition in_terms_target : Typing_env.t :=
  Eval vm_compute in
    match In_terms.join [Witness.r; Witness.x] In_terms.inputs with
    | Some (target, _) => target
    | None => Typing_env.create
    end.

Definition in_terms_analysis : Analysis.t unit :=
  Eval vm_compute in
    match In_terms.join [Witness.r; Witness.x] In_terms.inputs with
    | Some (_, analysis) => analysis
    | None => Analysis.empty
    end.

Lemma in_terms_join_result :
  In_terms.join [Witness.r; Witness.x] In_terms.inputs
  = Some (in_terms_target, in_terms_analysis).
Proof. vm_compute. reflexivity. Qed.

Theorem in_terms_use_not_admitted : in_terms_use_not_admitted_statement.
Proof.
  exists in_terms_target, in_terms_analysis.
  split; [ exact in_terms_join_result | ].
  intros Hadmits.
  destruct (in_terms_satisfied (Static_env [] [])) as [Huse Hsource].
  destruct (Hadmits _ _ Huse Hsource)
    as [valuation' [[_ [Htypes _]] [Hagrees _]]].
  assert (Hr : value_of_name valuation' (Name_var Witness.r)
               = Some (Value_const (Naked_immediate 0))).
  { cbn. rewrite (Hagrees Witness.r eq_refl). reflexivity. }
  assert (Hx : lookup Witness.x valuation' = None).
  { rewrite (Hagrees Witness.x eq_refl). reflexivity. }
  destruct (Htypes (Name_var Witness.r) _ _ _ eq_refl Hr)
    as [[[] _] | Hsatisfied].
  rewrite type_satisfied_naked_immediate in Hsatisfied.
  destruct Hsatisfied as [i [_ [_ Hrelations]]].
  destruct
    (Hrelations Relation.Is_null (Name_var Witness.x) (or_introl eq_refl))
    as [w [Hw _]].
  cbn in Hw. rewrite Hx in Hw. discriminate.
Qed.

Theorem repaired_join_not_strict : repaired_join_not_strict_statement.
Proof.
  intros Hsound.
  destruct in_terms_inputs_well_formed as [Hwell_formed Hkinded].
  destruct in_terms_use_not_admitted
    as [target [analysis [Hjoin Hnot_admitted]]].
  destruct (Hsound _ _ _ _ Hwell_formed Hkinded Hjoin)
    as [_ [_ [_ [_ [_ Hadmits]]]]].
  apply Hnot_admitted. apply (Hadmits 0 In_terms.use_env).
  exists tt. reflexivity.
Qed.

(* ======================================================================== *)
(* The three joins of discrepancy T5-10                                     *)
(* ======================================================================== *)

Theorem three_joins_compiler : three_joins_compiler_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem three_joins_repaired : three_joins_repaired_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

(* ======================================================================== *)
(* What the contract's premise on a run gives                               *)
(* ======================================================================== *)

Theorem equations_valued_joined_have_values
  : equations_valued_joined_have_values_statement.
Proof.
  intros cut_after scoped_envs s index scoped_env valuation Hcut Huse Hvalued
    x _ Heverywhere.
  apply joined_env_lookup in Huse. destruct Huse as [equations Hlookup].
  pose proof (Heverywhere index scoped_env equations Hlookup) as Hbound.
  destruct (cut_for_joins_found cut_after _ _ Hcut index scoped_env equations
              Hlookup)
    as [_ [_ [level [Hlevel ->]]]].
  unfold Keyed.is_bound in Hbound.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env_level.equations level)) as [ty | ] eqn:Hty;
    [ | discriminate ].
  apply (keyed_lookup_in name_eqb name_eqb_decides_equality) in Hty.
  destruct (cut_equations_member scoped_env cut_after level (Name_var x) ty
              Hlevel Hty)
    as [one_level [Hin [Hscope Hequation]]].
  destruct (Hvalued one_level (Name_var x) ty Hin Hscope Hequation)
    as [v Hv].
  exists v. exact Hv.
Qed.

(* ======================================================================== *)
(* The instance: what its environments hold                                 *)
(* ======================================================================== *)

Definition initial_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Brought_in.initial_tenv.

Definition first_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Brought_in.first_env.

Definition second_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Brought_in.second_env.

Lemma initial_names_to_types :
  Typing_env.names_to_types Brought_in.initial_tenv = initial_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma first_names_to_types :
  Typing_env.names_to_types Brought_in.first_env = first_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma second_names_to_types :
  Typing_env.names_to_types Brought_in.second_env = second_entries.
Proof. vm_compute. reflexivity. Qed.

(* The cases of a lookup in the entries of an environment of the
   instance: the variable is [x], [g], [p], or none of them. *)
Ltac entries_cases Hlookup n :=
  destruct n as [y | sym]; cbn [Keyed.lookup name_eqb fst snd] in Hlookup;
  [ destruct (Pos.eqb y 3%positive) eqn:Hx;
    [ apply Pos.eqb_eq in Hx; subst y
    | destruct (Pos.eqb y 2%positive) eqn:Hg;
      [ apply Pos.eqb_eq in Hg; subst y
      | destruct (Pos.eqb y 1%positive) eqn:Hp;
        [ apply Pos.eqb_eq in Hp; subst y | ] ] ]
  | ].

(* ======================================================================== *)
(* The inputs of the instance                                               *)
(* ======================================================================== *)

Lemma initial_invariant : typing_env_invariant Brought_in.initial_tenv.
Proof.
  apply add_variable_definition_invariant; [ | reflexivity ].
  apply add_variable_definition_invariant; [ | reflexivity ].
  exact create_invariant.
Qed.

Lemma initial_aliases_kinded : aliases_kinded Brought_in.initial_tenv.
Proof.
  apply add_variable_definition_aliases_kinded;
    [ vm_compute; reflexivity | reflexivity | ].
  apply add_variable_definition_aliases_kinded;
    [ vm_compute; reflexivity | reflexivity | ].
  exact create_aliases_kinded.
Qed.

Lemma initial_fits : typing_env_fits Brought_in.initial_tenv.
Proof.
  apply types_fit_b_fits;
    [ exact initial_invariant | exact initial_aliases_kinded | ].
  vm_compute. reflexivity.
Qed.

Lemma second_invariant : typing_env_invariant Brought_in.second_env.
Proof.
  apply replace_equation_invariant.
  - apply increment_scope_invariant. exact initial_invariant.
  - reflexivity.
  - intros y Hy. vm_compute in Hy. destruct Hy.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. vm_compute in Halias. discriminate.
  - left. vm_compute. reflexivity.
Qed.

Lemma second_aliases_kinded : aliases_kinded Brought_in.second_env.
Proof.
  apply replace_equation_aliases_kinded.
  - eexists. vm_compute. reflexivity.
  - apply increment_scope_aliases_kinded. exact initial_aliases_kinded.
Qed.

Lemma second_fits : typing_env_fits Brought_in.second_env.
Proof.
  apply types_fit_b_fits;
    [ exact second_invariant | exact second_aliases_kinded | ].
  vm_compute. reflexivity.
Qed.

(* The first joined environment, before [p] is given its type. *)
Definition first_env_before : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition
       (Typing_env.increment_scope Brought_in.initial_tenv) Brought_in.x Value
       Name_mode.in_types)
    (Name_var Brought_in.x) Brought_in.type_of_x.

Lemma first_before_invariant : typing_env_invariant first_env_before.
Proof.
  apply replace_equation_invariant.
  - apply add_variable_definition_invariant; [ | reflexivity ].
    apply increment_scope_invariant. exact initial_invariant.
  - reflexivity.
  - intros y Hy. vm_compute in Hy. destruct Hy as [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. vm_compute in Halias. discriminate.
  - left. vm_compute. reflexivity.
Qed.

Lemma first_invariant : typing_env_invariant Brought_in.first_env.
Proof.
  apply replace_equation_invariant.
  - exact first_before_invariant.
  - reflexivity.
  - intros y Hy. vm_compute in Hy. destruct Hy as [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. vm_compute in Halias. discriminate.
  - left. vm_compute. reflexivity.
Qed.

Lemma first_aliases_kinded : aliases_kinded Brought_in.first_env.
Proof.
  apply replace_equation_aliases_kinded.
  - eexists. vm_compute. reflexivity.
  - apply replace_equation_aliases_kinded.
    + eexists. vm_compute. reflexivity.
    + apply add_variable_definition_aliases_kinded;
        [ vm_compute; reflexivity | reflexivity | ].
      apply increment_scope_aliases_kinded. exact initial_aliases_kinded.
Qed.

Lemma first_fits : typing_env_fits Brought_in.first_env.
Proof.
  apply types_fit_b_fits;
    [ exact first_invariant | exact first_aliases_kinded | ].
  vm_compute. reflexivity.
Qed.

(* The names that the first environment knows. *)
Lemma initial_knows (n : name) :
  Typing_env.mem None Brought_in.initial_tenv n = true ->
  n = Name_var Brought_in.p ∨ n = Name_var Brought_in.g.
Proof.
  intros Hmem. unfold Typing_env.mem in Hmem.
  rewrite initial_names_to_types in Hmem. unfold initial_entries in Hmem.
  entries_cases Hmem n; try discriminate.
  - right. reflexivity.
  - left. reflexivity.
Qed.

Lemma first_scoped : scoped_env_of Brought_in.initial_tenv Brought_in.first_env.
Proof.
  split; [ exact first_fits | ]. split; [ | split ].
  - intros n Hn. destruct (initial_knows n Hn) as [-> | ->]; reflexivity.
  - intros n ty binding_time_and_mode Hlookup.
    rewrite initial_names_to_types in Hlookup.
    unfold initial_entries in Hlookup.
    entries_cases Hlookup n; try discriminate;
      injection Hlookup as <- <-; eexists;
      (split; [ vm_compute; reflexivity | reflexivity ]).
  - intros sym Hsym. discriminate Hsym.
Qed.

Lemma second_scoped :
  scoped_env_of Brought_in.initial_tenv Brought_in.second_env.
Proof.
  split; [ exact second_fits | ]. split; [ | split ].
  - intros n Hn. destruct (initial_knows n Hn) as [-> | ->]; reflexivity.
  - intros n ty binding_time_and_mode Hlookup.
    rewrite initial_names_to_types in Hlookup.
    unfold initial_entries in Hlookup.
    entries_cases Hlookup n; try discriminate;
      injection Hlookup as <- <-; eexists;
      (split; [ vm_compute; reflexivity | reflexivity ]).
  - intros sym Hsym. discriminate Hsym.
Qed.

(* The kinds of the [Simple]s of the two joined environments. *)
Definition kind_in_instance (s : simple) (kind : flambda_kind) : Prop :=
  match s with
  | Const c => FlambdaTypeGrammar.kind (type_for_const c) = kind
  | Var y =>
    (y = Brought_in.p ∧ kind = Value)
    ∨ (y = Brought_in.g ∧ kind = Naked_number Kind_naked_immediate)
    ∨ (y = Brought_in.x ∧ kind = Value)
  | Symbol _ => False
  end.

Lemma first_kinds (s : simple) (kind : flambda_kind) :
  simple_of_kind Brought_in.first_env s kind -> kind_in_instance s kind.
Proof.
  destruct s as [y | sym | c]; intros Hkind.
  - apply (simple_of_kind_name Brought_in.first_env (Name_var y) kind)
      in Hkind.
    rewrite first_names_to_types in Hkind. unfold first_entries in Hkind.
    destruct Hkind as [[ty [binding_time_and_mode [Hlookup <-]]]
                      | [Hlookup [Hmem _]]].
    + cbn [Keyed.lookup name_eqb fst snd] in Hlookup. cbn [kind_in_instance].
      destruct (Pos.eqb y 1%positive) eqn:Hp;
        [ apply Pos.eqb_eq in Hp; subst y; injection Hlookup as <- _;
          left; split; reflexivity | ].
      destruct (Pos.eqb y 3%positive) eqn:Hx;
        [ apply Pos.eqb_eq in Hx; subst y; injection Hlookup as <- _;
          right; right; split; reflexivity | ].
      destruct (Pos.eqb y 2%positive) eqn:Hg;
        [ apply Pos.eqb_eq in Hg; subst y; injection Hlookup as <- _;
          right; left; split; reflexivity | discriminate ].
    + unfold Typing_env.mem in Hmem. rewrite first_names_to_types in Hmem.
      unfold first_entries in Hmem. rewrite Hlookup in Hmem. discriminate.
  - apply (simple_of_kind_name Brought_in.first_env (Name_symbol sym) kind)
      in Hkind.
    destruct Hkind as [[ty [binding_time_and_mode [Hlookup _]]]
                      | [_ [Hmem _]]]; discriminate.
  - exact Hkind.
Qed.

Lemma second_kinds (s : simple) (kind : flambda_kind) :
  simple_of_kind Brought_in.second_env s kind -> kind_in_instance s kind.
Proof.
  destruct s as [y | sym | c]; intros Hkind.
  - apply (simple_of_kind_name Brought_in.second_env (Name_var y) kind)
      in Hkind.
    rewrite second_names_to_types in Hkind. unfold second_entries in Hkind.
    destruct Hkind as [[ty [binding_time_and_mode [Hlookup <-]]]
                      | [Hlookup [Hmem _]]].
    + cbn [Keyed.lookup name_eqb fst snd] in Hlookup. cbn [kind_in_instance].
      destruct (Pos.eqb y 1%positive) eqn:Hp;
        [ apply Pos.eqb_eq in Hp; subst y; injection Hlookup as <- _;
          left; split; reflexivity | ].
      destruct (Pos.eqb y 2%positive) eqn:Hg;
        [ apply Pos.eqb_eq in Hg; subst y; injection Hlookup as <- _;
          right; left; split; reflexivity | discriminate ].
    + unfold Typing_env.mem in Hmem. rewrite second_names_to_types in Hmem.
      unfold second_entries in Hmem. rewrite Hlookup in Hmem. discriminate.
  - apply (simple_of_kind_name Brought_in.second_env (Name_symbol sym) kind)
      in Hkind.
    destruct Hkind as [[ty [binding_time_and_mode [Hlookup _]]]
                      | [_ [Hmem _]]]; discriminate.
  - exact Hkind.
Qed.

Lemma kind_in_instance_unique (s : simple) (kind kind' : flambda_kind) :
  kind_in_instance s kind -> kind_in_instance s kind' -> kind = kind'.
Proof.
  destruct s as [y | sym | c]; cbn [kind_in_instance].
  - intros [[-> ->] | [[-> ->] | [-> ->]]] [[H ->] | [[H ->] | [H ->]]];
      try reflexivity; discriminate H.
  - intros [].
  - intros <- <-. reflexivity.
Qed.

Theorem brought_in_inputs : brought_in_inputs_statement.
Proof.
  destruct (meet_with_forgetful_join_has_contracts 20) as [Hkeeps Hsound].
  split; [ exact Hsound | ]. split; [ exact Hkeeps | ].
  split; [ exact initial_fits | ]. split.
  - intros scoped_env [<- | [<- | []]];
      [ exact first_scoped | exact second_scoped ].
  - intros scoped_env scoped_env' s kind kind' Hin Hin' Hkind Hkind'.
    apply (kind_in_instance_unique s).
    + destruct Hin as [<- | [<- | []]];
        [ exact (first_kinds _ _ Hkind) | exact (second_kinds _ _ Hkind) ].
    + destruct Hin' as [<- | [<- | []]];
        [ exact (first_kinds _ _ Hkind') | exact (second_kinds _ _ Hkind') ].
Qed.

(* ======================================================================== *)
(* The run of the instance                                                  *)
(* ======================================================================== *)

Lemma initial_aliases :
  Typing_env.aliases Brought_in.initial_tenv = Aliases.empty.
Proof. reflexivity. Qed.

Lemma second_aliases :
  Typing_env.aliases Brought_in.second_env = Aliases.empty.
Proof. reflexivity. Qed.

Theorem brought_in_run : brought_in_run_statement.
Proof.
  intros statics. split; [ | split; [ | split ] ].
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite initial_names_to_types in Hlookup.
      unfold initial_entries in Hlookup.
      entries_cases Hlookup n; try discriminate;
        injection Hlookup as <- _; injection Hvalue as <-; right.
      * rewrite type_satisfied_naked_immediate. reflexivity.
      * rewrite type_satisfied_value. reflexivity.
    + rewrite initial_aliases. apply empty_respected.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite second_names_to_types in Hlookup.
      unfold second_entries in Hlookup.
      entries_cases Hlookup n; try discriminate;
        injection Hlookup as <- _; injection Hvalue as <-; right.
      * rewrite type_satisfied_naked_immediate. reflexivity.
      * apply type_satisfied_b_iff. vm_compute. reflexivity.
    + rewrite second_aliases. apply empty_respected.
  - intros one_level n ty Hin Hscope Hequation.
    destruct Hin as [<- | [<- | []]].
    + vm_compute in Hequation. destruct Hequation as [Heq | []].
      injection Heq as <- _. eexists. reflexivity.
    + vm_compute in Hscope. discriminate Hscope.
  - intros y Hy. unfold in_scope_in_terms, Typing_env.mem in Hy.
    rewrite second_names_to_types in Hy. unfold second_entries in Hy.
    cbn [Keyed.lookup name_eqb fst snd] in Hy.
    destruct (Pos.eqb y 1%positive) eqn:Hp;
      [ apply Pos.eqb_eq in Hp; subst y; eexists; reflexivity | ].
    destruct (Pos.eqb y 2%positive) eqn:Hg;
      [ apply Pos.eqb_eq in Hg; subst y; eexists; reflexivity | ].
    discriminate Hy.
Qed.

(* ======================================================================== *)
(* What the join answers for the instance                                   *)
(* ======================================================================== *)

Definition compiler_result : Meet_env.t :=
  Eval vm_compute in
    match Brought_in.join Join_env.compiler_choices with
    | Some result => result
    | None => Brought_in.initial_env
    end.

Definition repaired_result : Meet_env.t :=
  Eval vm_compute in
    match Brought_in.join Join_env.repaired_choices with
    | Some result => result
    | None => Brought_in.initial_env
    end.

Lemma compiler_join_result :
  Brought_in.join Join_env.compiler_choices = Some compiler_result.
Proof. vm_compute. reflexivity. Qed.

Lemma repaired_join_result :
  Brought_in.join Join_env.repaired_choices = Some repaired_result.
Proof. vm_compute. reflexivity. Qed.

(* What is used of a result: its current level, at a scope after 0,
   has the equation of [x]; and the type that it has for [x] is that
   one. *)
Definition brings_in (result : Meet_env.t) : Prop :=
  let tenv := Meet_env.typing_env result in
  (0 < Typing_env.One_level.scope (Typing_env.current_level tenv))%Z
  ∧ (Name_var Brought_in.x, Brought_in.type_of_x)
    ∈ Typing_env_level.equations
        (Typing_env.One_level.level (Typing_env.current_level tenv))
  ∧ ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb (Name_var Brought_in.x)
        (Typing_env.names_to_types tenv)
      = Some (Brought_in.type_of_x, binding_time_and_mode).

Lemma compiler_brings_in : brings_in compiler_result.
Proof.
  split; [ reflexivity | ]. split.
  - vm_compute. right. left. reflexivity.
  - eexists. vm_compute. reflexivity.
Qed.

Lemma repaired_brings_in : brings_in repaired_result.
Proof.
  split; [ reflexivity | ]. split.
  - vm_compute. right. left. reflexivity.
  - eexists. vm_compute. reflexivity.
Qed.

(* The first environment has no equation after the scope 0. *)
Lemma initial_equations_valued (valuation : env) :
  equations_valued Brought_in.initial_tenv 0%Z valuation.
Proof.
  intros one_level n ty Hin Hscope Hequation.
  destruct Hin as [<- | []]. vm_compute in Hequation. destruct Hequation.
Qed.

Lemma brings_in_no_valuation (result : Meet_env.t) (statics : static_env)
    (valuation' : env) :
  brings_in result ->
  ~ (extended_from Brought_in.second_env Brought_in.valuation valuation'
     ∧ typing_env_satisfied_strictly statics valuation'
         (Meet_env.typing_env result)
     ∧ keeps_equations_valued Brought_in.initial_tenv
         (Meet_env.typing_env result) Brought_in.valuation valuation').
Proof.
  intros [Hscope [Hequation [binding_time_and_mode Hlookup]]]
    [Hfrom [[_ [Htypes _]] Hkeeps]].
  destruct (Hkeeps 0%Z (initial_equations_valued _) _ _ _
              (or_introl eq_refl) Hscope Hequation) as [v Hv].
  destruct (Htypes _ _ _ _ Hlookup Hv) as [[[] _] | Hsatisfied].
  assert (Hg : lookup Brought_in.g valuation'
               = Some (Value_const (Naked_immediate 7))).
  { rewrite (Hfrom Brought_in.g eq_refl). reflexivity. }
  unfold Brought_in.type_of_x in Hsatisfied.
  rewrite type_satisfied_value in Hsatisfied. cbn in Hsatisfied.
  rewrite head_of_kind_value_satisfied_eq in Hsatisfied.
  unfold variable_is in Hsatisfied. rewrite Hg in Hsatisfied.
  destruct Hsatisfied as [[_ Heq] | [_ [_ Heq]]]; discriminate Heq.
Qed.

Theorem brought_in_no_valuation : brought_in_no_valuation_statement.
Proof.
  intros choices [-> | ->].
  - exists compiler_result. split; [ exact compiler_join_result | ].
    intros statics valuation'.
    exact (brings_in_no_valuation _ _ _ compiler_brings_in).
  - exists repaired_result. split; [ exact repaired_join_result | ].
    intros statics valuation'.
    exact (brings_in_no_valuation _ _ _ repaired_brings_in).
Qed.

(* The types of an environment are true of the values that a
   valuation gives their names: a test. *)
Definition entries_satisfied_b (statics : static_env) (tenv : Typing_env.t)
    (valuation : env) : bool :=
  forallb
    (fun entry =>
       match value_of_name valuation (fst entry) with
       | None => true
       | Some v => type_satisfied_b statics valuation v (fst (snd entry))
       end)
    (Typing_env.names_to_types tenv).

Lemma entries_give_strict_satisfaction (statics : static_env)
    (tenv : Typing_env.t) (valuation : env) :
  Typing_env.is_bottom tenv = false ->
  entries_satisfied_b statics tenv valuation = true ->
  aliases_respected (Typing_env.aliases tenv) valuation ->
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  intros Hbottom Hentries Haliases.
  split; [ exact Hbottom | ]. split; [ | exact Haliases ].
  intros n ty binding_time_and_mode v Hlookup Hvalue. right.
  apply type_satisfied_b_iff.
  unfold entries_satisfied_b in Hentries. rewrite forallb_forall in Hentries.
  apply (keyed_lookup_in name_eqb name_eqb_decides_equality) in Hlookup.
  specialize (Hentries (n, (ty, binding_time_and_mode)) Hlookup).
  cbn [fst snd] in Hentries. rewrite Hvalue in Hentries. exact Hentries.
Qed.

Lemma valuation_of_result_extends :
  extended_from Brought_in.second_env Brought_in.valuation
    Brought_in.valuation_of_result.
Proof.
  intros y Hy. unfold Typing_env.mem in Hy.
  rewrite second_names_to_types in Hy. unfold second_entries in Hy.
  cbn [Keyed.lookup name_eqb fst snd] in Hy.
  destruct (Pos.eqb y 1%positive) eqn:Hp;
    [ apply Pos.eqb_eq in Hp; subst y; reflexivity | ].
  destruct (Pos.eqb y 2%positive) eqn:Hg;
    [ apply Pos.eqb_eq in Hg; subst y; reflexivity | ].
  discriminate Hy.
Qed.

Lemma compiler_result_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics Brought_in.valuation_of_result
    (Meet_env.typing_env compiler_result).
Proof.
  apply entries_give_strict_satisfaction.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - change (aliases_respected Aliases.empty Brought_in.valuation_of_result).
    apply empty_respected.
Qed.

Lemma repaired_result_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics Brought_in.valuation_of_result
    (Meet_env.typing_env repaired_result).
Proof.
  apply entries_give_strict_satisfaction.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - change (aliases_respected Aliases.empty Brought_in.valuation_of_result).
    apply empty_respected.
Qed.

Theorem brought_in_result_satisfied : brought_in_result_satisfied_statement.
Proof.
  intros choices [-> | ->].
  - exists compiler_result. split; [ exact compiler_join_result | ].
    split; [ exact valuation_of_result_extends | ].
    split; [ reflexivity | exact compiler_result_satisfied ].
  - exists repaired_result. split; [ exact repaired_join_result | ].
    split; [ exact valuation_of_result_extends | ].
    split; [ reflexivity | exact repaired_result_satisfied ].
Qed.

Theorem model_join_scoped_envs_not_sound
  : model_join_scoped_envs_not_sound_statement.
Proof.
  intros choices Hchoices Hsound.
  destruct brought_in_inputs
    as [Hmeet_sound [Hmeet_keeps [Hfits [Hscoped Hagree]]]].
  destruct (brought_in_no_valuation choices Hchoices)
    as [result [Hjoin Hnone]].
  destruct (Hsound Brought_in.meet_head Brought_in.initial_env result
              [Brought_in.first_env; Brought_in.second_env] Hmeet_sound
              Hmeet_keeps Hfits Hscoped Hagree Hjoin)
    as [Hvalues _].
  destruct (brought_in_run (Static_env [] []))
    as [Hsat_initial [Hsat_second [Hvalued _]]].
  destruct (Hvalues Brought_in.second_env (Static_env [] [])
              Brought_in.valuation (or_intror (or_introl eq_refl))
              (Forall_nil _) Hsat_initial Hsat_second Hvalued)
    as [valuation' Hwitness].
  exact (Hnone (Static_env [] []) valuation' Hwitness).
Qed.
