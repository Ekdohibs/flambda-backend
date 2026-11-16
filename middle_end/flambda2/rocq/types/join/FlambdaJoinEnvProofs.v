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


(* Proofs of the statements of FlambdaJoinEnvSpec.v. Nothing in this
   file needs to be reviewed. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypeAliasKindsSpec FlambdaTypeAliasKindsProofs
  FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The instance: what its environments hold                                 *)
(* ======================================================================== *)

Definition source_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Witness.source_env.

Definition use_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Witness.use_env.

Lemma source_names_to_types :
  Typing_env.names_to_types Witness.source_env = source_entries.
Proof. vm_compute. reflexivity. Qed.

Lemma use_names_to_types :
  Typing_env.names_to_types Witness.use_env = use_entries.
Proof. vm_compute. reflexivity. Qed.

Definition use_null_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types Witness.use_env_null.

Lemma use_null_names_to_types :
  Typing_env.names_to_types Witness.use_env_null = use_null_entries.
Proof. vm_compute. reflexivity. Qed.

(* A use of the instance is one of its two use environments. *)
Lemma witness_use_at (index : Index.t) (use : Typing_env.t) :
  use_at Witness.inputs index use ->
  use = Witness.use_env ∨ use = Witness.use_env_null.
Proof.
  intros [id Hnth]. cbn in Hnth.
  destruct index as [ | [ | index]]; cbn in Hnth.
  - injection Hnth as _ <-. left. reflexivity.
  - injection Hnth as _ <-. right. reflexivity.
  - destruct index; discriminate.
Qed.

(* The stand-in for the join of types recognises one type. *)
Lemma is_type_of_x_eq (t : flambda_type) :
  Witness.is_type_of_x t = true -> t = Witness.type_of_x.
Proof.
  destruct t as [ty | ty | k ty | ty | ty]; try discriminate.
  destruct ty as [ | d | ]; try discriminate.
  destruct d as [head | s]; try discriminate.
  destruct head as [non_null is_null].
  destruct non_null; try discriminate.
  destruct is_null as [ | is_null]; try discriminate.
  destruct is_null as [r' | ]; try discriminate.
  cbn. destruct r'; intros Heq; try discriminate. reflexivity.
Qed.

Lemma is_type_of_x_null_eq (t : flambda_type) :
  Witness.is_type_of_x_null t = true -> t = Witness.type_of_x_null.
Proof.
  destruct t as [ty | ty | k ty | ty | ty]; try discriminate.
  destruct ty as [ | d | ]; try discriminate.
  destruct d as [head | s]; try discriminate.
  destruct head as [non_null is_null].
  destruct non_null; try discriminate.
  destruct is_null as [ | is_null]; try discriminate.
  destruct is_null as [r' | ]; try discriminate.
  cbn. destruct r'; intros Heq; try discriminate. reflexivity.
Qed.

Theorem witness_join_of_types_upper_bound
  : witness_join_of_types_upper_bound_statement.
Proof.
  intros s s' ts ty index t_index statics valuation valuation' v Hjoin Hin
    Hr Hsatisfied.
  unfold Witness.join_of_types in Hjoin.
  destruct ts as [ | binding ts]; [ discriminate | ].
  destruct (forallb _ (binding :: ts)) eqn:Hall; [ | discriminate ].
  injection Hjoin as <- <-. split; [ reflexivity | ].
  rewrite forallb_forall in Hall. specialize (Hall _ Hin). cbn in Hall.
  apply orb_true_iff in Hall. destruct Hall as [Hall | Hall].
  - apply is_type_of_x_eq in Hall. subst t_index.
    unfold Witness.type_of_x in *.
    rewrite type_satisfied_value in *. cbn in *.
    rewrite head_of_kind_value_satisfied_eq in *.
    unfold variable_is in *. rewrite Hr. exact Hsatisfied.
  - apply is_type_of_x_null_eq in Hall. subst t_index.
    unfold Witness.type_of_x, Witness.type_of_x_null in *.
    rewrite type_satisfied_value in *. cbn in *.
    rewrite head_of_kind_value_satisfied_eq in *.
    unfold variable_is in *. rewrite Hr.
    destruct Hsatisfied as [Hnull | [_ [[] _]]]. left. exact Hnull.
Qed.

(* ======================================================================== *)
(* The inputs of the instance                                               *)
(* ======================================================================== *)

Lemma source_invariant : typing_env_invariant Witness.source_env.
Proof.
  apply add_variable_definition_invariant; [ | reflexivity ].
  apply add_variable_definition_invariant; [ | reflexivity ].
  exact create_invariant.
Qed.

Lemma use_invariant : typing_env_invariant Witness.use_env.
Proof.
  apply replace_equation_invariant.
  - apply increment_scope_invariant. exact source_invariant.
  - reflexivity.
  - intros y [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. discriminate.
  - left. reflexivity.
Qed.

Lemma use_null_invariant : typing_env_invariant Witness.use_env_null.
Proof.
  apply replace_equation_invariant.
  - apply increment_scope_invariant. exact source_invariant.
  - reflexivity.
  - intros y [<- | []]. reflexivity.
  - intros ty' binding_time_and_mode Hlookup. vm_compute in Hlookup.
    injection Hlookup as <- _. reflexivity.
  - intros alias Halias. discriminate.
  - left. reflexivity.
Qed.

(* What an environment of the instance has for a name: the cases of a
   lookup in its entries. *)
Ltac entries_cases Hlookup n :=
  destruct n as [y | sym]; cbn [Keyed.lookup name_eqb fst snd] in Hlookup;
  [ destruct (Pos.eqb y 2%positive) eqn:Hx;
    [ apply Pos.eqb_eq in Hx; subst y
    | destruct (Pos.eqb y 1%positive) eqn:Hr;
      [ apply Pos.eqb_eq in Hr; subst y | ] ]
  | ].

Lemma source_no_bottom_types : no_bottom_types Witness.source_env.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite source_names_to_types in Hlookup. unfold source_entries in Hlookup.
  entries_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma use_no_bottom_types : no_bottom_types Witness.use_env.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite use_names_to_types in Hlookup. unfold use_entries in Hlookup.
  entries_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma source_kinded : typing_env_kinded Witness.source_env.
Proof.
  apply add_variable_definition_kinded; [ reflexivity | reflexivity | ].
  apply add_variable_definition_kinded; [ reflexivity | reflexivity | ].
  exact create_kinded.
Qed.

Lemma use_kinded : typing_env_kinded Witness.use_env.
Proof.
  apply replace_equation_kinded.
  - eexists. vm_compute. reflexivity.
  - intros simple kind [Heq | []]. injection Heq as <- <-.
    eexists. vm_compute. reflexivity.
  - apply increment_scope_kinded. exact source_kinded.
Qed.

Lemma use_null_no_bottom_types : no_bottom_types Witness.use_env_null.
Proof.
  intros _ n ty binding_time_and_mode Hlookup.
  rewrite use_null_names_to_types in Hlookup.
  unfold use_null_entries in Hlookup.
  entries_cases Hlookup n; try discriminate;
    injection Hlookup as <- _; reflexivity.
Qed.

Lemma use_null_kinded : typing_env_kinded Witness.use_env_null.
Proof.
  apply replace_equation_kinded.
  - eexists. vm_compute. reflexivity.
  - intros simple kind [Heq | []]. injection Heq as <- <-.
    eexists. vm_compute. reflexivity.
  - apply increment_scope_kinded. exact source_kinded.
Qed.

Theorem witness_inputs_well_formed : witness_inputs_well_formed_statement.
Proof.
  split; [ split; [ | split; [ | split; [ | split ] ] ] | ].
  - split; [ exact source_invariant | ].
    intros index use Huse.
    destruct (witness_use_at _ _ Huse); subst use;
      [ exact use_invariant | exact use_null_invariant ].
  - split; [ exact source_no_bottom_types | ].
    intros index use Huse.
    destruct (witness_use_at _ _ Huse); subst use;
      [ exact use_no_bottom_types | exact use_null_no_bottom_types ].
  - intros index use Huse _.
    destruct (witness_use_at _ _ Huse); subst use;
      eexists; vm_compute; reflexivity.
  - intros index use sym Huse Hdefined.
    destruct (witness_use_at _ _ Huse); subst use; discriminate.
  - split.
    + intros y Hknown. unfold knows, Typing_env.mem in Hknown.
      change (source Witness.inputs) with Witness.source_env in Hknown.
      rewrite source_names_to_types in Hknown.
      unfold source_entries in Hknown.
      cbn [Keyed.lookup name_eqb fst snd] in Hknown.
      destruct (Pos.eqb y 2%positive) eqn:Hx;
        [ apply Pos.eqb_eq in Hx; subst y; reflexivity | ].
      destruct (Pos.eqb y 1%positive) eqn:Hr;
        [ apply Pos.eqb_eq in Hr; subst y; reflexivity | discriminate ].
    + intros index use y Huse Hknown.
      unfold knows, Typing_env.mem in Hknown.
      destruct (witness_use_at _ _ Huse); subst use;
        [ rewrite use_names_to_types in Hknown; unfold use_entries in Hknown
        | rewrite use_null_names_to_types in Hknown;
          unfold use_null_entries in Hknown ];
        cbn [Keyed.lookup name_eqb fst snd] in Hknown;
        (destruct (Pos.eqb y 2%positive) eqn:Hx;
           [ apply Pos.eqb_eq in Hx; subst y; reflexivity | ]);
        (destruct (Pos.eqb y 1%positive) eqn:Hr;
           [ apply Pos.eqb_eq in Hr; subst y; reflexivity | discriminate ]).
  - intros index use Huse.
    destruct (witness_use_at _ _ Huse); subst use;
      [ exact use_kinded | exact use_null_kinded ].
Qed.

(* ======================================================================== *)
(* The run of the instance                                                  *)
(* ======================================================================== *)

Lemma source_aliases : Typing_env.aliases Witness.source_env = Aliases.empty.
Proof. reflexivity. Qed.

Lemma use_aliases : Typing_env.aliases Witness.use_env = Aliases.empty.
Proof. reflexivity. Qed.

Theorem witness_run_satisfies : witness_run_satisfies_statement.
Proof.
  intros statics. split.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite use_names_to_types in Hlookup. unfold use_entries in Hlookup.
      entries_cases Hlookup n; try discriminate.
      injection Hlookup as <- _. injection Hvalue as <-.
      right. rewrite type_satisfied_naked_immediate. reflexivity.
    + rewrite use_aliases. apply empty_respected.
  - split; [ reflexivity | ]. split.
    + intros n ty binding_time_and_mode v Hlookup Hvalue.
      rewrite source_names_to_types in Hlookup.
      unfold source_entries in Hlookup.
      entries_cases Hlookup n; try discriminate.
      injection Hlookup as <- _. injection Hvalue as <-.
      right. rewrite type_satisfied_naked_immediate. reflexivity.
    + rewrite source_aliases. apply empty_respected.
Qed.

(* ======================================================================== *)
(* What the join answers for the instance                                   *)
(* ======================================================================== *)

Definition witness_target : Typing_env.t :=
  Eval vm_compute in
    match Witness.join [Witness.r; Witness.x] Witness.inputs with
    | Some (target, _) => target
    | None => Typing_env.create
    end.

Definition witness_analysis : Analysis.t unit :=
  Eval vm_compute in
    match Witness.join [Witness.r; Witness.x] Witness.inputs with
    | Some (_, analysis) => analysis
    | None => Analysis.empty
    end.

Lemma witness_join_result :
  Witness.join [Witness.r; Witness.x] Witness.inputs
  = Some (witness_target, witness_analysis).
Proof. vm_compute. reflexivity. Qed.

Theorem witness_use_not_admitted : witness_use_not_admitted_statement.
Proof.
  exists witness_target, witness_analysis.
  split; [ exact witness_join_result | ]. split; [ | split ].
  - intros Hadmits.
    destruct (witness_run_satisfies (Static_env [] [])) as [Huse Hsource].
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
  - intros Hscrutinees.
    destruct
      (Hscrutinees Witness.r Witness.x Relation.Is_null _ _ eq_refl
         (or_introl eq_refl)
         (or_introl (conj (or_introl eq_refl) (ex_intro _ _ eq_refl))))
      as [[_ [v Hv]] | [simples [kind [simple [v [Hdefinition _]]]]]];
      discriminate.
  - intros Hjoined.
    destruct (Hjoined Witness.x eq_refl) as [v Hv]; [ | discriminate ].
    intros index use Huse _.
    destruct (witness_use_at _ _ Huse); subst use;
      eexists; eexists; split; vm_compute; reflexivity.
Qed.

Theorem transcribed_join_not_strict : transcribed_join_not_strict_statement.
Proof.
  intros Hsound.
  destruct witness_inputs_well_formed as [Hwell_formed Hkinded].
  destruct witness_use_not_admitted
    as [target [analysis [Hjoin [Hnot_admitted _]]]].
  destruct (Hsound _ _ _ _ Hwell_formed Hkinded Hjoin)
    as [_ [_ [_ [_ [_ Hadmits]]]]].
  apply Hnot_admitted. apply (Hadmits 0 Witness.use_env).
  exists tt. reflexivity.
Qed.

(* ======================================================================== *)
(* Where the rounds record a relation                                       *)
(* ======================================================================== *)

Lemma recorded_add (relations : Join_env.inverse_relations)
    (var result : variable) (relation relation' : Relation.t)
    (scrutinee scrutinee' : name) :
  relation_recorded
    (Join_env.add_to_inverse_relations relations var relation' scrutinee')
    result relation scrutinee ->
  relation_recorded relations result relation scrutinee
  ∨ scrutinee = scrutinee'.
Proof.
  intros [recorded [Hlookup Hin]].
  unfold Join_env.add_to_inverse_relations in Hlookup.
  cbn [lookup] in Hlookup.
  destruct (Pos.eqb result var) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst var. injection Hlookup as <-.
    destruct Hin as [Hhead | Hin].
    + injection Hhead as _ <-. right. reflexivity.
    + destruct (lookup result relations) as [existing | ] eqn:Hexisting.
      * left. exists existing. split; [ exact Hexisting | exact Hin ].
      * destruct Hin.
  - left. exists recorded. split; assumption.
Qed.

Theorem recovered_relations_on_name : recovered_relations_on_name_statement.
Proof.
  intros relations relations' n ty ty' result relation scrutinee Hrecover
    Hrecorded.
  unfold Join_env.recover_inverse_relations in Hrecover.
  repeat
    (match type of Hrecover with
     | context [match ?e with _ => _ end] => destruct e eqn:?
     end; try discriminate);
    injection Hrecover as _ <-;
    first
      [ left; exact Hrecorded
      | exact (recorded_add _ _ _ _ _ _ _ Hrecorded) ].
Qed.

Lemma keeps_definitions_refl (s : Join_env.t) : keeps_definitions s s.
Proof.
  split; [ intros index; reflexivity | intros x definition H; exact H ].
Qed.

Lemma keeps_definitions_trans (s1 s2 s3 : Join_env.t) :
  keeps_definitions s1 s2 -> keeps_definitions s2 s3 ->
  keeps_definitions s1 s3.
Proof.
  intros [Henvs Hdefs] [Henvs' Hdefs']. split.
  - intros index. rewrite Henvs'. apply Henvs.
  - intros x definition H. apply Hdefs'. apply Hdefs. exact H.
Qed.

Lemma add_equation_keeps (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  keeps_definitions s s'.
Proof.
  unfold Join_env.add_equation_in_target_env.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  intros H. injection H as <-. split; [ intros index; reflexivity | ].
  intros y definition Hy. exact Hy.
Qed.

Lemma recorded_where_chosen_same (choices : Join_env.choices)
    (s s' : Join_env.t) (relations : Join_env.inverse_relations) :
  recorded_where_chosen choices s s' relations relations.
Proof. intros result relation scrutinee H. left. exact H. Qed.

Lemma recorded_where_chosen_trans (choices : Join_env.choices)
    (s1 s2 s3 : Join_env.t)
    (relations1 relations2 relations3 : Join_env.inverse_relations) :
  keeps_definitions s1 s2 ->
  keeps_definitions s2 s3 ->
  recorded_where_chosen choices s1 s2 relations1 relations2 ->
  recorded_where_chosen choices s2 s3 relations2 relations3 ->
  recorded_where_chosen choices s1 s3 relations1 relations3.
Proof.
  intros H12 H23 Hfirst Hsecond result relation scrutinee Hrecorded.
  destruct (Hsecond _ _ _ Hrecorded)
    as [Hrecorded2 | [x [types [s0 [Heq [Hbefore [Hafter Htest]]]]]]].
  - destruct (Hfirst _ _ _ Hrecorded2)
      as [Hrecorded1 | [x [types [s0 [Heq [Hbefore [Hafter Htest]]]]]]].
    + left. exact Hrecorded1.
    + right. exists x, types, s0. split; [ exact Heq | ].
      split; [ exact Hbefore | ]. split; [ | exact Htest ].
      exact (keeps_definitions_trans _ _ _ Hafter H23).
  - right. exists x, types, s0. split; [ exact Heq | ].
    split; [ | split; [ exact Hafter | exact Htest ] ].
    exact (keeps_definitions_trans _ _ _ H12 Hbefore).
Qed.

Theorem round_records_where_chosen : round_records_where_chosen_statement.
Proof.
  intros choices jt s s' equations relations relations' Hjt Hround.
  unfold Join_env.n_way_join_round in Hround.
  match type of Hround with
  | List.fold_left ?f _ _ = _ => set (step := f) in Hround
  end.
  assert (Hnone : ∀ bindings, List.fold_left step bindings None = None).
  { induction bindings as [ | binding bindings IH];
      [ reflexivity | exact IH ]. }
  revert relations s Hround. generalize (compact equations) as bindings.
  induction bindings as [ | [var types] bindings IH];
    intros relations s Hround.
  - cbn in Hround. injection Hround as <- <-.
    split; [ apply keeps_definitions_refl | apply recorded_where_chosen_same ].
  - cbn [List.fold_left] in Hround.
    destruct (step (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite Hnone in Hround; discriminate ].
    destruct (IH _ _ Hround) as [Hkeeps Hchosen].
    assert (Hfirst :
              keeps_definitions s s1
              ∧ recorded_where_chosen choices s s1 relations relations1).
    { unfold step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      pose proof (Hjt _ _ _ _ Hjoin) as Hkeeps0.
      destruct joined as [ty | ].
      - destruct (Join_env.recover_relations_on choices s0 var types)
          eqn:Htest.
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as <- <-.
          pose proof (add_equation_keeps _ _ _ _ Hadd) as Hkeeps2.
          split; [ exact (keeps_definitions_trans _ _ _ Hkeeps0 Hkeeps2) | ].
          intros result relation scrutinee Hrecorded.
          destruct (recovered_relations_on_name _ _ _ _ _ _ _ _ Hrecover
                      Hrecorded) as [Hbefore | ->].
          * left. exact Hbefore.
          * right. exists var, types, s0. split; [ reflexivity | ].
            split; [ exact Hkeeps0 | ]. split; [ exact Hkeeps2 | exact Htest ].
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as <- <-.
          pose proof (add_equation_keeps _ _ _ _ Hadd) as Hkeeps2.
          split; [ exact (keeps_definitions_trans _ _ _ Hkeeps0 Hkeeps2) | ].
          apply recorded_where_chosen_same.
      - injection Hstep as <- <-. split; [ exact Hkeeps0 | ].
        apply recorded_where_chosen_same. }
    destruct Hfirst as [Hkeeps1 Hchosen1].
    split; [ exact (keeps_definitions_trans _ _ _ Hkeeps1 Hkeeps) | ].
    exact (recorded_where_chosen_trans _ _ _ _ _ _ _ Hkeeps1 Hkeeps Hchosen1
             Hchosen).
Qed.

Theorem delayed_equations_record_where_chosen
  : delayed_equations_record_where_chosen_statement.
Proof.
  intros fuel choices jt s s' equations relations relations' Hjt.
  revert s equations relations.
  induction fuel as [ | fuel IH]; intros s equations relations Hloop;
    [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as <- <-.
    split; [ apply keeps_definitions_refl | apply recorded_where_chosen_same ].
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    destruct (round_records_where_chosen _ _ _ _ _ _ _ Hjt Hround)
      as [Hkeeps1 Hchosen1].
    destruct (IH _ _ _ Hloop) as [Hkeeps Hchosen].
    split; [ exact (keeps_definitions_trans _ _ _ Hkeeps1 Hkeeps) | ].
    exact (recorded_where_chosen_trans _ _ _ _ _ _ _ Hkeeps1 Hkeeps Hchosen1
             Hchosen).
Qed.

Theorem loop_records_where_chosen : loop_records_where_chosen_statement.
Proof.
  intros depth p fuel choices jt s s' existentials relations relations' Hjt.
  revert s existentials relations.
  induction depth as [ | depth IH]; intros s existentials relations Hloop.
  - injection Hloop as <- <-.
    split; [ apply keeps_definitions_refl | apply recorded_where_chosen_same ].
  - cbn [Join_env.n_way_join_loop] in Hloop.
    destruct (Keyed.is_empty existentials).
    + injection Hloop as <- <-.
      split;
        [ apply keeps_definitions_refl | apply recorded_where_chosen_same ].
    + destruct (Join_env.equations_of_existentials p s (compact existentials))
        as [equations | ]; [ | discriminate ].
      destruct
        (Join_env.n_way_join_delayed_equations fuel choices jt s relations
           equations)
        as [[relations1 s1] | ] eqn:Hrounds; [ | discriminate ].
      destruct
        (delayed_equations_record_where_chosen _ _ _ _ _ _ _ _ Hjt Hrounds)
        as [Hkeeps1 Hchosen1].
      destruct (IH _ _ _ Hloop) as [Hkeeps Hchosen].
      split; [ exact (keeps_definitions_trans _ _ _ Hkeeps1 Hkeeps) | ].
      exact (recorded_where_chosen_trans _ _ _ _ _ _ _ Hkeeps1 Hkeeps Hchosen1
               Hchosen).
Qed.

(* ======================================================================== *)
(* The repaired test                                                        *)
(* ======================================================================== *)

Lemma index_equal_decides : decides_equality Index.equal.
Proof. intros index index'. apply Nat.eqb_eq. Qed.

(* An index that has a joined environment is among the keys. *)
Lemma joined_env_key (envs : Joined_envs.t) (index : Index.t)
    (tenv : Typing_env.t) :
  Joined_envs.get_nth_joined_env envs index = Some tenv ->
  index ∈ Joined_envs.keys envs.
Proof.
  unfold Joined_envs.get_nth_joined_env, Joined_envs.keys. intros Hnth.
  apply (keyed_among_in _ Index.equal index_equal_decides).
  rewrite (keyed_among_domain _ Index.equal index_equal_decides).
  apply keyed_is_bound_lookup.
  destruct (Keyed.lookup Index.equal index envs) as [binding | ];
    [ | discriminate ].
  exists binding. reflexivity.
Qed.

(* A variable that has a canonical simple at a name mode is bound. *)
Lemma canonical_exn_known (tenv : Typing_env.t)
    (min_name_mode : option Name_mode.t) (x : variable)
    (answer : option simple) :
  Typing_env.get_canonical_simple_exn tenv min_name_mode None (Var x)
  = Some answer ->
  knows tenv x.
Proof.
  unfold Typing_env.get_canonical_simple_exn,
    Typing_env.binding_time_and_mode_of_simple,
    Typing_env.binding_time_and_mode,
    Typing_env.find_with_binding_time_and_mode,
    Typing_env.find_with_binding_time_and_mode', knows, Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)) as [[ty time] | ];
    [ reflexivity | discriminate ].
Qed.

Theorem repaired_test_gives_values : repaired_test_gives_values_statement.
Proof.
  intros s x types index tenv exempt statics valuation Htest Hindex
    Hsatisfied Hbound.
  cbn in Htest. apply andb_true_iff in Htest. destruct Htest as [_ Htest].
  unfold Join_env.in_terms_in_all_joined_envs in Htest.
  rewrite forallb_forall in Htest.
  unfold Join_env.joined_env in Hindex.
  specialize (Htest index (joined_env_key _ _ _ Hindex)).
  rewrite Hindex in Htest.
  destruct (Join_env.stands_for s index x) as [simple | ] eqn:Hstands;
    [ | discriminate ].
  unfold Join_env.has_alias_in_terms in Htest.
  destruct (Typing_env.get_canonical_simple_exn tenv (Some Name_mode.normal)
              None simple) as [[canonical | ] | ] eqn:Hcanonical;
    try discriminate.
  assert (Hvalue : ∃ v : value, eval_simple valuation simple = Some v).
  { rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsatisfied
                  Hcanonical).
    destruct canonical as [y | sym | c].
    - exact (Hbound y Htest).
    - eexists. reflexivity.
    - eexists. reflexivity. }
  destruct Hvalue as [v Hv].
  unfold Join_env.stands_for in Hstands.
  destruct (lookup x (Join_env.definitions_of_existentials s))
    as [[simples kind] | ] eqn:Hdefinition.
  - right. exists simples, kind, simple, v.
    split; [ exact Hdefinition | split; [ exact Hstands | exact Hv ] ].
  - injection Hstands as <-. left. split; [ | exists v; exact Hv ].
    exact (canonical_exn_known _ _ _ _ Hcanonical).
Qed.

(* A variable that has a value by the definitions of a state has one by
   those of a later state. *)
Lemma has_value_at_use_keeps (known : variable -> Prop) (s s' : Join_env.t)
    (index : Index.t) (valuation : env) (x : variable) :
  keeps_definitions s s' ->
  has_value_at_use known (Join_env.definitions_of_existentials s) index
    valuation x ->
  has_value_at_use known (Join_env.definitions_of_existentials s') index
    valuation x.
Proof.
  intros [_ Hdefinitions]
    [Hvalue | [simples [kind [simple [v [Hdefinition [Hfind Hv]]]]]]].
  - left. exact Hvalue.
  - right. exists simples, kind, simple, v.
    split; [ exact (Hdefinitions _ _ Hdefinition) | ].
    split; [ exact Hfind | exact Hv ].
Qed.

Theorem repaired_rounds_scrutinees_have_values
  : repaired_rounds_scrutinees_have_values_statement.
Proof.
  intros depth p fuel jt s s1 s2 equations existentials relations1 relations2
    result relation scrutinee index tenv exempt statics valuation Hjt
    Hrounds Hloop Hrecorded Hindex Hsatisfied Hbound.
  destruct (delayed_equations_record_where_chosen _ _ _ _ _ _ _ _ Hjt Hrounds)
    as [Hkeeps1 Hchosen1].
  destruct (loop_records_where_chosen _ _ _ _ _ _ _ _ _ _ Hjt Hloop)
    as [Hkeeps2 Hchosen2].
  pose proof (recorded_where_chosen_trans _ _ _ _ _ _ _ Hkeeps1 Hkeeps2
                Hchosen1 Hchosen2) as Hchosen.
  destruct (Hchosen _ _ _ Hrecorded)
    as [[recorded [Hlookup _]]
       | [x [types [s0 [Heq [Hbefore [Hafter Htest]]]]]]];
    [ discriminate | ].
  exists x. split; [ exact Heq | ].
  apply (has_value_at_use_keeps _ s0 s2 _ _ _ Hafter).
  apply (repaired_test_gives_values s0 x types index tenv exempt statics
           valuation Htest); try assumption.
  destruct Hbefore as [Henvs _]. rewrite Henvs. exact Hindex.
Qed.

(* ======================================================================== *)
(* The repaired join on the instance                                        *)
(* ======================================================================== *)

Definition repaired_target : Typing_env.t :=
  Eval vm_compute in
    match Witness.repaired_join [Witness.r; Witness.x] Witness.inputs with
    | Some (target, _) => target
    | None => Typing_env.create
    end.

Definition repaired_analysis : Analysis.t unit :=
  Eval vm_compute in
    match Witness.repaired_join [Witness.r; Witness.x] Witness.inputs with
    | Some (_, analysis) => analysis
    | None => Analysis.empty
    end.

Lemma repaired_join_result :
  Witness.repaired_join [Witness.r; Witness.x] Witness.inputs
  = Some (repaired_target, repaired_analysis).
Proof. vm_compute. reflexivity. Qed.

Definition repaired_entries : Keyed.finite_map name _ :=
  Eval vm_compute in Typing_env.names_to_types repaired_target.

Lemma repaired_names_to_types :
  Typing_env.names_to_types repaired_target = repaired_entries.
Proof. vm_compute. reflexivity. Qed.

Theorem witness_repaired_join_admits : witness_repaired_join_admits_statement.
Proof.
  exists repaired_target, repaired_analysis.
  split; [ exact repaired_join_result | ].
  intros index use Huse statics valuation [_ [Htypes _]] _.
  exists valuation.
  split;
    [ | split;
        [ intros y _; reflexivity
        | split;
          [ intros y _; reflexivity
          | split; intros y simples kind simple v Hlookup; discriminate ] ] ].
  split; [ reflexivity | ]. split; [ | apply empty_respected ].
  intros n ty binding_time_and_mode v Hlookup Hvalue.
  rewrite repaired_names_to_types in Hlookup.
  unfold repaired_entries in Hlookup.
  entries_cases Hlookup n; try discriminate; injection Hlookup as <- _;
    right.
  - destruct (witness_use_at _ _ Huse); subst use;
      destruct (Htypes (Name_var 2%positive) _ _ v eq_refl Hvalue)
        as [[[] _] | Hsatisfied].
    + exact Hsatisfied.
    + rewrite type_satisfied_value in *. cbn in *.
      rewrite head_of_kind_value_satisfied_eq in *.
      destruct Hsatisfied as [Hnull | [_ [[] _]]]. left. exact Hnull.
  - destruct (witness_use_at _ _ Huse); subst use;
      destruct (Htypes (Name_var 1%positive) _ _ v eq_refl Hvalue)
        as [[[] _] | Hsatisfied];
      exact Hsatisfied.
Qed.

(* ======================================================================== *)
(* The state of a join: its readings and its invariant                      *)
(* ======================================================================== *)

Theorem kind_in_target_env_exists : kind_in_target_env_exists_statement.
Proof.
  intros s x.
  unfold exists_in_target_env, kind_in_target_env,
    Bindings_in_target_env.exists_in_target_env,
    Source_env.exists_in_source_env, Typing_env.mem, is_bound.
  destruct (Keyed.lookup name_eqb (Name_var x) _) as [[ty time] | ];
    [ reflexivity | ].
  cbn [orb].
  destruct (lookup x (Bindings_in_target_env.existential_variables _));
    [ reflexivity | ].
  cbn [orb].
  destruct (lookup x (Bindings_in_target_env.imported_variables _));
    reflexivity.
Qed.

Theorem created_state_valid : created_state_valid_statement.
Proof.
  intros avoid source envs Hsource Henvs.
  split; [ | split; [ | split ] ].
  - split; [ exact Hsource | split; [ exact Henvs | ] ].
    intros x Hx. discriminate.
  - intros x Hx. discriminate.
  - split; [ | split ].
    + intros x simples kind Hx. discriminate.
    + intros simples x Hx. discriminate.
    + intros x Hx. discriminate.
  - split.
    + intros x Hx. discriminate.
    + intros x Hx. discriminate.
Qed.

Definition with_definitions (s : Join_env.t) (index : Index.t)
    (valuation : env) : env :=
  tabulate (value_of_definition s index valuation)
    (domain (Join_env.definitions_of_existentials s))
  ++ valuation.

Lemma lookup_with_definitions (s : Join_env.t) (index : Index.t)
    (valuation : env) (x : variable) :
  lookup x (with_definitions s index valuation)
  = match value_of_definition s index valuation x with
    | Some v => Some v
    | None => lookup x valuation
    end.
Proof.
  unfold with_definitions. rewrite lookup_app, lookup_tabulate, among_domain.
  unfold value_of_definition, is_bound.
  destruct (lookup x (Join_env.definitions_of_existentials s))
    as [[simples kind] | ]; reflexivity.
Qed.

Theorem state_valid_has_witnesses : state_valid_has_witnesses_statement.
Proof.
  intros s index use statics valuation
    [_ [Hnew [[Hrecorded _] _]]] Hindex _.
  exists (with_definitions s index valuation), use.
  split; [ exact Hindex | ]. split.
  - intros x Hknown. rewrite lookup_with_definitions.
    unfold value_of_definition.
    destruct (lookup x (Join_env.definitions_of_existentials s))
      as [[simples kind] | ] eqn:Hdefinition; [ | reflexivity ].
    exfalso.
    assert (Hexistential : is_existential s x).
    { unfold is_existential. apply is_bound_lookup. exists kind.
      exact (Hrecorded _ _ _ Hdefinition). }
    destruct (Hnew x Hexistential) as [_ Hunknown].
    exact (Hunknown _ _ Hindex Hknown).
  - intros x simples kind simple v Hdefinition Hfind Hv.
    rewrite lookup_with_definitions. unfold value_of_definition.
    rewrite Hdefinition, Hfind, Hv. reflexivity.
Qed.

(* ---------------------------------------------------------------------- *)
(* Growth                                                                  *)
(* ---------------------------------------------------------------------- *)

Abbreviation grows :=
  (state_grows Join_env.joined_env Join_env.definitions_of_existentials
     kind_in_target_env).

Abbreviation exists_in_target := (exists_in_target_env kind_in_target_env).

Lemma grows_refl (s : Join_env.t) : grows s s.
Proof.
  split; [ intros index; reflexivity | ].
  split; [ intros x definition H; exact H | intros x kind H; exact H ].
Qed.

Lemma grows_trans (s1 s2 s3 : Join_env.t) :
  grows s1 s2 -> grows s2 s3 -> grows s1 s3.
Proof.
  intros [Henvs [Hdefs Hkinds]] [Henvs' [Hdefs' Hkinds']].
  split; [ intros index; rewrite Henvs'; apply Henvs | ].
  split.
  - intros x definition H. apply Hdefs'. apply Hdefs. exact H.
  - intros x kind H. apply Hkinds'. apply Hkinds. exact H.
Qed.

Lemma grows_exists (s s' : Join_env.t) (x : variable) :
  grows s s' -> exists_in_target s x = true -> exists_in_target s' x = true.
Proof.
  intros [_ [_ Hkinds]]. unfold exists_in_target_env.
  destruct (kind_in_target_env s x) as [kind | ] eqn:Hkind;
    [ | discriminate ].
  rewrite (Hkinds _ _ Hkind). reflexivity.
Qed.

(* Two states with the same readings. *)
Definition same_readings (s s' : Join_env.t) : Prop :=
  Join_env.joined_envs s' = Join_env.joined_envs s
  ∧ Join_env.bindings s' = Join_env.bindings s
  ∧ Join_env.definitions_of_existentials s'
    = Join_env.definitions_of_existentials s
  ∧ Join_env.imported_vars s' = Join_env.imported_vars s.

Lemma same_readings_valid (s s' : Join_env.t) :
  same_readings s s' -> state_valid s -> state_valid s'.
Proof.
  destruct s, s'. unfold same_readings. cbn.
  intros [-> [-> [-> ->]]] Hvalid. exact Hvalid.
Qed.

Lemma same_readings_grows (s s' : Join_env.t) :
  same_readings s s' -> grows s s'.
Proof.
  destruct s, s'. unfold same_readings. cbn.
  intros [-> [-> [-> ->]]].
  split; [ intros index; reflexivity | ].
  split; [ intros x definition H; exact H | intros x kind H; exact H ].
Qed.

Lemma add_equation_same_readings (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' -> same_readings s s'.
Proof.
  unfold Join_env.add_equation_in_target_env.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  intros H. injection H as <-. repeat split.
Qed.

Lemma add_equations_same_readings (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  same_readings s (Join_env.add_equations_in_joined_envs s x types).
Proof. repeat split. Qed.

(* ---------------------------------------------------------------------- *)
(* Vectors of simples                                                      *)
(* ---------------------------------------------------------------------- *)

Lemma simples_equal_find (simples simples' : Simples_in_joined_envs.t) :
  Simples_in_joined_envs.equal simples simples' = true ->
  ∀ index : Index.t,
    Simples_in_joined_envs.find index simples
    = Simples_in_joined_envs.find index simples'.
Proof.
  unfold Simples_in_joined_envs.equal. intros Hequal index.
  apply andb_true_iff in Hequal. destruct Hequal as [Hforward Hbackward].
  rewrite (keyed_for_all_bindings_b _ Index.equal index_equal_decides)
    in Hforward, Hbackward.
  rewrite (keyed_for_all_bindings_lookup _ Index.equal index_equal_decides)
    in Hforward, Hbackward.
  unfold Simples_in_joined_envs.find in *.
  destruct (Keyed.lookup Index.equal index simples) as [simple | ] eqn:Hfind.
  - specialize (Hforward _ _ Hfind). cbn beta in Hforward.
    destruct (Keyed.lookup Index.equal index simples') as [simple' | ];
      [ | discriminate ].
    apply simple_equal_decides_equality in Hforward. subst simple'.
    reflexivity.
  - destruct (Keyed.lookup Index.equal index simples') as [simple' | ]
      eqn:Hfind'; [ | reflexivity ].
    specialize (Hbackward _ _ Hfind'). cbn beta in Hbackward.
    unfold Keyed.is_bound in Hbackward. rewrite Hfind in Hbackward.
    discriminate.
Qed.

Lemma simples_equal_refl (simples : Simples_in_joined_envs.t) :
  Simples_in_joined_envs.equal simples simples = true.
Proof.
  unfold Simples_in_joined_envs.equal. apply andb_true_iff. split;
    rewrite (keyed_for_all_bindings_b _ Index.equal index_equal_decides);
    rewrite (keyed_for_all_bindings_lookup _ Index.equal index_equal_decides);
    intros index simple Hfind.
  - unfold Simples_in_joined_envs.find. rewrite Hfind.
    apply simple_equal_decides_equality. reflexivity.
  - unfold Keyed.is_bound. rewrite Hfind. reflexivity.
Qed.

Lemma existential_keeps (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  fresh_avoids p ->
  state_valid s ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  state_valid s'
  ∧ grows s s'
  ∧ is_existential s' x
  ∧ ∃ (simples' : Simples_in_joined_envs.t) (kind' : flambda_kind),
      lookup x (Join_env.definitions_of_existentials s')
      = Some (simples', kind')
      ∧ Simples_in_joined_envs.equal canonicals simples' = true.
Proof.
  intros Hfresh Hvalid Hmake. pose proof Hvalid as Hvalid0.
  destruct Hvalid
    as [[Hsource [Huses Hexistentials]]
          [Hnew [[Hrecorded [Hvectors Hdefined_all]] [Himported Himports]]]].
  unfold Join_env.existential_for_these_canonicals,
    Bindings_in_target_env.existential_for_these_simples in Hmake.
  destruct (Bindings_in_target_env.has_existential_for_these_simples
              (Join_env.bindings s) canonicals) as [existing | ] eqn:Hhas.
  - destruct (Hvectors _ _ Hhas) as [simples' [kind' [Hdefinition Hequal]]].
    assert (Hbound : is_bound existing (Join_env.definitions_of_existentials s)
                     = true).
    { apply is_bound_lookup. eexists. exact Hdefinition. }
    rewrite Hbound in Hmake. injection Hmake as <- <-.
    match goal with
    | |- state_valid ?s' ∧ _ =>
      assert (Hsame : same_readings s s') by (repeat split)
    end.
    split; [ exact (same_readings_valid _ _ Hsame Hvalid0) | ].
    split; [ exact (same_readings_grows _ _ Hsame) | ].
    split.
    + unfold is_existential. cbn. apply is_bound_lookup. exists kind'.
      exact (Hrecorded _ _ _ Hdefinition).
    + exists simples', kind'. split; [ exact Hdefinition | exact Hequal ].
  - set (x0 :=
           Join_env.fresh p
             (Bindings_in_target_env.names_to_avoid (Join_env.bindings s)))
      in *.
    assert (Hx0 : ~ x0 ∈ names_to_avoid s) by apply Hfresh.
    assert (Hunbound :
              is_bound x0 (Join_env.definitions_of_existentials s) = false).
    { destruct (is_bound x0 (Join_env.definitions_of_existentials s))
        eqn:Hb; [ | reflexivity ].
      exfalso. apply is_bound_lookup in Hb. destruct Hb as [[simples k] Hd].
      apply Hx0. apply Hexistentials. unfold is_existential.
      apply is_bound_lookup. exists k. exact (Hrecorded _ _ _ Hd). }
    clearbody x0. cbv beta iota zeta in Hmake.
    rewrite Hunbound in Hmake. injection Hmake as <- <-.
    assert (Hother :
              ∀ y : variable,
                is_existential s y
                ∨ knows (Join_env.source_env s) y
                ∨ (∃ (index : Index.t) (use : Typing_env.t),
                     Join_env.joined_env s index = Some use ∧ knows use y) ->
                Pos.eqb y x0 = false).
    { intros y Hy. apply Pos.eqb_neq. intros ->. apply Hx0.
      destruct Hy as [Hy | [Hy | [index [use [Hindex Hy]]]]];
        [ exact (Hexistentials _ Hy) | exact (Hsource _ Hy)
        | exact (Huses _ _ _ Hindex Hy) ]. }
    assert (Hdefined :
              ∀ (y : variable) (simples : Simples_in_joined_envs.t)
                (k : flambda_kind),
                lookup y (Join_env.definitions_of_existentials s)
                = Some (simples, k) ->
                Pos.eqb y x0 = false).
    { intros y simples k Hy. apply Hother. left. unfold is_existential.
      apply is_bound_lookup. exists k. exact (Hrecorded _ _ _ Hy). }
    split; [ | split; [ | split ] ].
    + split; [ | split; [ | split ] ].
      * split; [ intros y Hy; right; exact (Hsource _ Hy) | ].
        split;
          [ intros index use y Hindex Hy; right;
            exact (Huses _ _ _ Hindex Hy) | ].
        intros y Hy. unfold is_existential, is_bound in Hy.
        cbn [Join_env.bindings Bindings_in_target_env.existential_variables
               lookup] in Hy.
        destruct (Pos.eqb y x0) eqn:Heq.
        -- left. apply Pos.eqb_eq in Heq. symmetry. exact Heq.
        -- right. apply Hexistentials. exact Hy.
      * intros y Hy. unfold is_existential, is_bound in Hy.
        cbn [Join_env.bindings Bindings_in_target_env.existential_variables
               lookup] in Hy.
        destruct (Pos.eqb y x0) eqn:Heq.
        -- apply Pos.eqb_eq in Heq. subst y. split.
           ++ intros Hknown. apply Hx0. exact (Hsource _ Hknown).
           ++ intros index use Hindex Hknown. apply Hx0.
              exact (Huses _ _ _ Hindex Hknown).
        -- exact (Hnew y Hy).
      * split; [ | split ].
        -- intros y simples k Hy.
           cbn [Join_env.bindings Join_env.definitions_of_existentials
                  Bindings_in_target_env.existential_variables lookup]
             in Hy |- *.
           destruct (Pos.eqb y x0) eqn:Heq.
           ++ injection Hy as _ <-. reflexivity.
           ++ exact (Hrecorded _ _ _ Hy).
        -- intros simples y Hy.
           unfold Bindings_in_target_env.has_existential_for_these_simples
             in Hy.
           cbn [Join_env.bindings
                  Bindings_in_target_env.existential_for_these_simples_map
                  Keyed.lookup] in Hy.
           cbn [Join_env.definitions_of_existentials lookup].
           destruct (Simples_in_joined_envs.equal simples canonicals)
             eqn:Hequal.
           ++ injection Hy as <-. exists canonicals, kind.
              rewrite Pos.eqb_refl. split; [ reflexivity | exact Hequal ].
           ++ destruct (Hvectors _ _ Hy)
                as [simples' [k [Hdefinition Hequal']]].
              exists simples', k. rewrite (Hdefined _ _ _ Hdefinition).
              split; [ exact Hdefinition | exact Hequal' ].
        -- intros y Hy. unfold is_existential, is_bound in Hy |- *.
           cbn [Join_env.bindings Join_env.definitions_of_existentials
                  Bindings_in_target_env.existential_variables lookup]
             in Hy |- *.
           destruct (Pos.eqb y x0); [ reflexivity | ].
           exact (Hdefined_all y Hy).
      * split; [ exact Himported | exact Himports ].
    + split; [ intros index; reflexivity | split ].
      * intros y [simples k] Hy.
        cbn [Join_env.definitions_of_existentials lookup].
        rewrite (Hdefined _ _ _ Hy). exact Hy.
      * intros y k Hy. unfold kind_in_target_env in *.
        cbn [Join_env.bindings Bindings_in_target_env.source_env
               Bindings_in_target_env.existential_variables
               Bindings_in_target_env.imported_variables lookup].
        destruct (Keyed.lookup name_eqb (Name_var y) _) as [[ty time] | ];
          [ exact Hy | ].
        destruct (lookup y
                    (Bindings_in_target_env.existential_variables
                       (Join_env.bindings s))) as [k' | ] eqn:Hev.
        -- rewrite (Hother y); [ exact Hy | ].
           left. unfold is_existential. apply is_bound_lookup. exists k'.
           exact Hev.
        -- rewrite (Hother y); [ exact Hy | ].
           right. right.
           assert (Hy' : is_imported s y).
           { unfold is_imported. apply is_bound_lookup. exists k. exact Hy. }
           exact (proj2 (Himports _ Hy')).
    + unfold is_existential, is_bound.
      cbn [Join_env.bindings Bindings_in_target_env.existential_variables
             lookup].
      rewrite Pos.eqb_refl. reflexivity.
    + exists canonicals, kind.
      cbn [Join_env.definitions_of_existentials lookup].
      rewrite Pos.eqb_refl. split; [ reflexivity | apply simples_equal_refl ].
Qed.

(* ---------------------------------------------------------------------- *)
(* Importing                                                               *)
(* ---------------------------------------------------------------------- *)

(* What is shown of a function that imports a variable. *)
Definition import_keeps
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s ->
    import_var s x = Some s' ->
    state_valid s' ∧ grows s s' ∧ exists_in_target s' x = true.

Lemma import_vars_keeps
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    state_valid s'
    ∧ grows s s'
    ∧ ∀ x : variable, x ∈ vars -> exists_in_target s' x = true.
Proof.
  intros Himport.
  induction vars as [ | var vars IH]; intros s s' Hvalid Hfold.
  - injection Hfold as <-.
    split; [ exact Hvalid | split; [ apply grows_refl | intros x [] ] ].
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Himport _ _ _ Hvalid Hvar) as [Hvalid1 [Hgrows1 Hexists1]].
      destruct (IH _ _ Hvalid1 Hfold) as [Hvalid' [Hgrows' Hexists']].
      split; [ exact Hvalid' | ].
      split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows') | ].
      intros x [<- | Hin];
        [ exact (grows_exists _ _ _ Hgrows' Hexists1)
        | exact (Hexists' x Hin) ].
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma import_type_with_keeps
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (ty : flambda_type) :
  import_keeps import_var ->
  state_valid s ->
  Join_env.import_type_transitive_with import_var s ty = Some s' ->
  state_valid s' ∧ grows s s' ∧ type_in_target kind_in_target_env s' ty.
Proof.
  intros Himport Hvalid Himported.
  exact (import_vars_keeps import_var (free_variables ty) Himport s s' Hvalid
           Himported).
Qed.

Lemma add_alias_keeps (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  state_valid s ->
  state_valid (Join_env.add_alias_of_existential p s kind var canonicals)
  ∧ grows s (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hfresh Hvalid. unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
    as [Hvalid1 [Hgrows1 _]].
  match goal with
  | |- state_valid ?s' ∧ _ =>
    assert (Hsame : same_readings s1 s') by (repeat split)
  end.
  split; [ exact (same_readings_valid _ _ Hsame Hvalid1) | ].
  exact (grows_trans _ _ _ Hgrows1 (same_readings_grows _ _ Hsame)).
Qed.

Lemma add_equation_keeps_valid (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  state_valid s ->
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  state_valid s' ∧ grows s s'.
Proof.
  intros Hvalid Hadd.
  pose proof (add_equation_same_readings _ _ _ _ Hadd) as Hsame.
  split;
    [ exact (same_readings_valid _ _ Hsame Hvalid)
    | exact (same_readings_grows _ _ Hsame) ].
Qed.

Lemma add_equals_keeps (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  state_valid s ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  state_valid s' ∧ grows s s'.
Proof.
  intros Hfresh Himport Hvalid Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_keeps_valid _ _ _ _ Hvalid Hadd).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    destruct (Himport _ _ _ Hvalid Hvar) as [Hvalid1 [Hgrows1 _]].
    destruct (add_equation_keeps_valid _ _ _ _ Hvalid1 Hadd)
      as [Hvalid' Hgrows'].
    split; [ exact Hvalid' | exact (grows_trans _ _ _ Hgrows1 Hgrows') ].
  - injection Hadd as <-. exact (add_alias_keeps _ _ _ _ _ Hfresh Hvalid).
Qed.

Lemma concrete_keeps (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps import_var ->
  state_valid s ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  state_valid s' ∧ grows s s'.
Proof.
  intros Himport Hvalid Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    destruct (add_equation_keeps_valid _ _ _ _ Hvalid Hadd)
      as [Hvalid1 Hgrows1].
    destruct (import_type_with_keeps _ _ _ _ Himport Hvalid1 Hconcrete)
      as [Hvalid' [Hgrows' _]].
    split; [ exact Hvalid' | exact (grows_trans _ _ _ Hgrows1 Hgrows') ].
  - injection Hconcrete as <-.
    pose proof (add_equations_same_readings s var types) as Hsame.
    split;
      [ exact (same_readings_valid _ _ Hsame Hvalid)
      | exact (same_readings_grows _ _ Hsame) ].
Qed.

Lemma kind_of_var_in_known
    (envs :
       list (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
    (x : variable) (kind : flambda_kind) :
  Joined_envs.kind_of_var_in envs x = Some kind ->
  ∃ (index : Index.t) (env : Typing_env.t)
    (equations : Keyed.finite_map name flambda_type),
    (index, (env, equations)) ∈ envs ∧ knows env x.
Proof.
  induction envs as [ | [index [env equations]] envs IH];
    cbn [Joined_envs.kind_of_var_in]; [ discriminate | ].
  assert (Hrest :
            Joined_envs.kind_of_var_in envs x = Some kind ->
            ∃ (index0 : Index.t) (env0 : Typing_env.t)
              (equations0 : Keyed.finite_map name flambda_type),
              (index0, (env0, equations0)) ∈ (index, (env, equations)) :: envs
              ∧ knows env0 x).
  { intros H. destruct (IH H) as [i [e [eqs [Hin Hknown]]]].
    exists i, e, eqs. split; [ right; exact Hin | exact Hknown ]. }
  destruct (Typing_env.find env (Name_var x) None) as [ty | ];
    [ | exact Hrest ].
  destruct (Typing_env.mem None env (Name_var x)) eqn:Hmem; [ | exact Hrest ].
  intros _. exists index, env, equations.
  split; [ left; reflexivity | exact Hmem ].
Qed.

Lemma kind_of_imported_var_known (s : Join_env.t) (x : variable)
    (kind : flambda_kind) :
  Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x = Some kind ->
  ∃ (index : Index.t) (use : Typing_env.t),
    Join_env.joined_env s index = Some use ∧ knows use x.
Proof.
  intros H.
  destruct (kind_of_var_in_known _ _ _ H)
    as [index [env [equations [Hin Hknown]]]].
  apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
  exists index, env. split; [ | exact Hknown ].
  unfold Join_env.joined_env, Joined_envs.get_nth_joined_env.
  rewrite Hin. reflexivity.
Qed.

(* The state in which a variable has just been brought in. *)
Definition with_import (s : Join_env.t) (x : variable) (kind : flambda_kind)
    : Join_env.t :=
  {| Join_env.joined_envs := Join_env.joined_envs s;
     Join_env.types_in_target_env := Join_env.types_in_target_env s;
     Join_env.types_in_joined_envs := Join_env.types_in_joined_envs s;
     Join_env.aliases_of_existentials := Join_env.aliases_of_existentials s;
     Join_env.imported_vars := x :: Join_env.imported_vars s;
     Join_env.definitions_of_existentials :=
       Join_env.definitions_of_existentials s;
     Join_env.bindings :=
       Bindings_in_target_env.import_from_all_envs (Join_env.bindings s) x
         kind |}.

Lemma with_import_keeps (s : Join_env.t) (x : variable) (kind : flambda_kind) :
  state_valid s ->
  ~ knows (Join_env.source_env s) x ->
  (∃ (index : Index.t) (use : Typing_env.t),
     Join_env.joined_env s index = Some use ∧ knows use x) ->
  state_valid (with_import s x kind)
  ∧ grows s (with_import s x kind)
  ∧ exists_in_target (with_import s x kind) x = true.
Proof.
  intros [Havoided [Hnew [Hrecorded [Himported Himports]]]] Hsource Huse.
  unfold with_import, Bindings_in_target_env.import_from_all_envs.
  destruct (is_bound x
              (Bindings_in_target_env.imported_variables
                 (Join_env.bindings s)))
    eqn:Hbound.
  - split; [ | split ].
    + split; [ exact Havoided | ]. split; [ exact Hnew | ].
      split; [ exact Hrecorded | ]. split; [ | exact Himports ].
      intros y Hy. cbn [Join_env.imported_vars] in Hy.
      rewrite among_cons in Hy. apply orb_true_iff in Hy.
      destruct Hy as [Hy | Hy]; [ | exact (Himported y Hy) ].
      apply Pos.eqb_eq in Hy. subst y. exact Hbound.
    + split; [ intros index; reflexivity | ].
      split; [ intros y definition H; exact H | intros y k H; exact H ].
    + rewrite kind_in_target_env_exists.
      unfold Bindings_in_target_env.exists_in_target_env.
      cbn [Join_env.bindings]. rewrite Hbound. apply orb_true_r.
  - assert (Hstep :
              ∀ y : variable,
                is_imported
                  {| Join_env.joined_envs := Join_env.joined_envs s;
                     Join_env.types_in_target_env :=
                       Join_env.types_in_target_env s;
                     Join_env.types_in_joined_envs :=
                       Join_env.types_in_joined_envs s;
                     Join_env.aliases_of_existentials :=
                       Join_env.aliases_of_existentials s;
                     Join_env.imported_vars := x :: Join_env.imported_vars s;
                     Join_env.definitions_of_existentials :=
                       Join_env.definitions_of_existentials s;
                     Join_env.bindings :=
                       {| Bindings_in_target_env.source_env :=
                            Bindings_in_target_env.source_env
                              (Join_env.bindings s);
                          Bindings_in_target_env
                          .existential_for_these_simples_map :=
                            Bindings_in_target_env
                            .existential_for_these_simples_map
                              (Join_env.bindings s);
                          Bindings_in_target_env.imported_variables :=
                            (x, kind)
                            :: Bindings_in_target_env.imported_variables
                                 (Join_env.bindings s);
                          Bindings_in_target_env.existential_variables :=
                            Bindings_in_target_env.existential_variables
                              (Join_env.bindings s);
                          Bindings_in_target_env.names_to_avoid :=
                            Bindings_in_target_env.names_to_avoid
                              (Join_env.bindings s) |} |} y
                <-> y = x ∨ is_imported s y).
    { intros y. unfold is_imported, is_bound.
      cbn [Join_env.bindings Bindings_in_target_env.imported_variables lookup].
      destruct (Pos.eqb y x) eqn:Heq.
      - apply Pos.eqb_eq in Heq. split; [ intros _; left; exact Heq | auto ].
      - apply Pos.eqb_neq in Heq.
        split; [ intros H; right; exact H | ].
        intros [H | H]; [ contradiction | exact H ]. }
    split; [ | split ].
    + split; [ exact Havoided | ]. split; [ exact Hnew | ].
      split; [ exact Hrecorded | ]. split.
      * intros y Hy. cbn [Join_env.imported_vars] in Hy. apply Hstep.
        rewrite among_cons in Hy. apply orb_true_iff in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply Pos.eqb_eq; exact Hy | right; exact (Himported y Hy) ].
      * intros y Hy. apply Hstep in Hy. destruct Hy as [-> | Hy];
          [ split; [ exact Hsource | exact Huse ] | exact (Himports y Hy) ].
    + split; [ intros index; reflexivity | ].
      split; [ intros y definition H; exact H | ].
      intros y k Hy. unfold kind_in_target_env in *.
      cbn [Join_env.bindings Bindings_in_target_env.source_env
             Bindings_in_target_env.existential_variables
             Bindings_in_target_env.imported_variables lookup].
      destruct (Keyed.lookup name_eqb (Name_var y) _) as [[ty time] | ];
        [ exact Hy | ].
      destruct (lookup y
                  (Bindings_in_target_env.existential_variables
                     (Join_env.bindings s))) as [k' | ]; [ exact Hy | ].
      destruct (Pos.eqb y x) eqn:Heq; [ | exact Hy ].
      apply Pos.eqb_eq in Heq. subst y. unfold is_bound in Hbound.
      rewrite Hy in Hbound. discriminate.
    + rewrite kind_in_target_env_exists.
      unfold Bindings_in_target_env.exists_in_target_env, is_bound.
      cbn [Join_env.bindings Bindings_in_target_env.imported_variables lookup].
      rewrite Pos.eqb_refl. apply orb_true_r.
Qed.

Lemma import_body_keeps (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hfresh Himport s s' x Hvalid Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource.
  { injection Hbody as <-.
    split; [ exact Hvalid | split; [ apply grows_refl | ] ].
    rewrite kind_in_target_env_exists.
    unfold Bindings_in_target_env.exists_in_target_env.
    unfold Join_env.source_env in Hsource. rewrite Hsource. reflexivity. }
  destruct (among x (Join_env.imported_vars s)) eqn:Hamong.
  { injection Hbody as <-.
    split; [ exact Hvalid | split; [ apply grows_refl | ] ].
    rewrite kind_in_target_env_exists.
    unfold Bindings_in_target_env.exists_in_target_env.
    destruct Hvalid as [_ [_ [_ [Himported _]]]].
    pose proof (Himported x Hamong) as Hbound. unfold is_imported in Hbound.
    rewrite Hbound. apply orb_true_r. }
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ] eqn:Hkind; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hunknown : ~ knows (Join_env.source_env s) x).
  { unfold knows. unfold Source_env.exists_in_source_env in Hsource.
    rewrite Hsource. discriminate. }
  destruct (with_import_keeps s x kind Hvalid Hunknown
              (kind_of_imported_var_known _ _ _ Hkind))
    as [Hvalid1 [Hgrows1 Hexists1]].
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ];
    [ | discriminate ].
  assert (Hrest : state_valid s' ∧ grows (with_import s x kind) s').
  { destruct (existsb _ _).
    - destruct
        (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
           _ _) as [[canonical_simples | ] | ]; try discriminate.
      exact (add_equals_keeps _ _ _ _ _ _ _ Hfresh Himport Hvalid1 Hbody).
    - exact (concrete_keeps _ _ _ _ _ _ Himport Hvalid1 Hbody). }
  destruct Hrest as [Hvalid' Hgrows'].
  split; [ exact Hvalid' | ].
  split; [ exact (grows_trans _ _ _ Hgrows1 Hgrows') | ].
  exact (grows_exists _ _ _ Hgrows' Hexists1).
Qed.

Lemma import_var_keeps (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p -> import_keeps (Join_env.import_var_transitive p fuel).
Proof.
  intros Hfresh. induction fuel as [ | fuel IH].
  - intros s s' x _ H. discriminate.
  - exact (import_body_keeps p _ Hfresh IH).
Qed.

(* ---------------------------------------------------------------------- *)
(* The join of simples                                                     *)
(* ---------------------------------------------------------------------- *)

Lemma keyed_lookup_of_in {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K) (a : A)
    (bindings : list (K * A)) :
  NoDup (map fst bindings) ->
  (key, a) ∈ bindings ->
  Keyed.lookup eqb key bindings = Some a.
Proof.
  induction bindings as [ | [bound b] bindings IH]; intros Hnodup Hin;
    [ destruct Hin | ].
  cbn [map fst] in Hnodup. apply NoDup_cons_iff in Hnodup.
  destruct Hnodup as [Hfresh Hnodup]. cbn [Keyed.lookup].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. rewrite (keyed_eqb_refl eqb Heqb). reflexivity.
  - destruct (eqb key bound) eqn:Htest; [ | exact (IH Hnodup Hin) ].
    apply Heqb in Htest. subst bound. exfalso. apply Hfresh.
    apply in_map_iff. exists (key, a). split; [ reflexivity | exact Hin ].
Qed.

(* The vector of canonical simples that the join of simples makes. *)
Lemma canonical_simples_fold (envs : Joined_envs.t) :
  ∀ (simples : list (Index.t * simple))
    (acc canonicals : list (Index.t * simple)),
    List.fold_left
      (fun acc binding =>
         let* acc := acc in
         let* env := Joined_envs.get_nth_joined_env envs (fst binding) in
         Some
           ((fst binding,
             Typing_env.get_canonical_simple_ignoring_name_mode env
               (snd binding))
            :: acc))
      simples (Some acc)
    = Some canonicals ->
    map fst canonicals = rev (map fst simples) ++ map fst acc
    ∧ ∀ (index : Index.t) (simple : simple),
        (index, simple) ∈ simples ->
        ∃ env : Typing_env.t,
          Joined_envs.get_nth_joined_env envs index = Some env
          ∧ (index,
             Typing_env.get_canonical_simple_ignoring_name_mode env simple)
            ∈ canonicals.
Proof.
  induction simples as [ | [index0 simple0] simples IH];
    intros acc canonicals Hfold.
  - injection Hfold as <-. split; [ reflexivity | intros index simple [] ].
  - cbn [List.fold_left fst snd] in Hfold.
    destruct (Joined_envs.get_nth_joined_env envs index0) as [env0 | ]
      eqn:Hnth.
    + destruct (IH _ _ Hfold) as [Hkeys Hin]. split.
      * rewrite Hkeys. cbn [map fst rev]. rewrite <- app_assoc. reflexivity.
      * intros index simple [Heq | Hsimple]; [ | exact (Hin _ _ Hsimple) ].
        injection Heq as <- <-. exists env0. split; [ exact Hnth | ].
        clear -Hfold.
        assert (Hkept :
                  ∀ (simples : list (Index.t * simple))
                    (acc canonicals : list (Index.t * simple))
                    (binding : Index.t * simple),
                    List.fold_left
                      (fun acc binding =>
                         let* acc := acc in
                         let* env :=
                           Joined_envs.get_nth_joined_env envs (fst binding)
                         in
                         Some
                           ((fst binding,
                             Typing_env.get_canonical_simple_ignoring_name_mode
                               env (snd binding))
                            :: acc))
                      simples (Some acc)
                    = Some canonicals ->
                    binding ∈ acc -> binding ∈ canonicals).
        { clear. induction simples as [ | [index1 simple1] simples IH];
            intros acc canonicals binding Hfold Hin.
          - injection Hfold as <-. exact Hin.
          - cbn [List.fold_left fst snd] in Hfold.
            destruct (Joined_envs.get_nth_joined_env envs index1) as [env1 | ].
            + apply (IH _ _ _ Hfold). right. exact Hin.
            + exfalso. clear -Hfold.
              induction simples as [ | b simples IH];
                [ discriminate | exact (IH Hfold) ]. }
        apply (Hkept _ _ _ _ Hfold). left. reflexivity.
    + exfalso. clear -Hfold.
      induction simples as [ | b simples IH];
        [ discriminate | exact (IH Hfold) ].
Qed.

Lemma canonical_simples_find (envs : Joined_envs.t)
    (simples : list (Index.t * simple)) (canonicals : Simples_in_joined_envs.t)
    (index : Index.t) (simple : simple) :
  Joined_envs.get_canonical_simples_ignoring_name_mode envs simples
  = Some canonicals ->
  NoDup (map fst simples) ->
  (index, simple) ∈ simples ->
  ∃ env : Typing_env.t,
    Joined_envs.get_nth_joined_env envs index = Some env
    ∧ Simples_in_joined_envs.find index canonicals
      = Some (Typing_env.get_canonical_simple_ignoring_name_mode env simple).
Proof.
  intros Hcanonicals Hnodup Hin.
  destruct (canonical_simples_fold envs _ _ _ Hcanonicals) as [Hkeys Hfound].
  destruct (Hfound _ _ Hin) as [env [Hnth Hbinding]].
  exists env. split; [ exact Hnth | ].
  apply (keyed_lookup_of_in Index.equal index_equal_decides);
    [ | exact Hbinding ].
  rewrite Hkeys, app_nil_r. apply NoDup_rev. exact Hnodup.
Qed.

Lemma source_var_of_some (source : Typing_env.t) (simple : simple)
    (var : variable) :
  Source_env.source_var_of source simple = Some var ->
  simple = Var var ∧ Source_env.exists_in_source_env source var = true.
Proof.
  destruct simple as [x | sym | c]; cbn [Source_env.source_var_of];
    try discriminate.
  destruct (Source_env.exists_in_source_env source x) eqn:Hexists;
    [ | discriminate ].
  intros H. injection H as <-. split; [ reflexivity | exact Hexists ].
Qed.

Lemma latest_bound_cases (source : Typing_env.t) (var var0 w : variable) :
  Source_env.latest_bound source var var0 = Some w -> w = var ∨ w = var0.
Proof.
  unfold Source_env.latest_bound.
  destruct (Pos.eqb var var0);
    [ intros H; injection H as <-; left; reflexivity | ].
  destruct (Typing_env.binding_time_and_mode source (Name_var var));
    [ | discriminate ].
  destruct (Typing_env.binding_time_and_mode source (Name_var var0));
    [ | discriminate ].
  destruct (Binding_time.strictly_earlier _ _);
    [ intros H; injection H as <-; right; reflexivity | ].
  destruct (Binding_time.strictly_earlier _ _);
    [ intros H; injection H as <-; left; reflexivity | discriminate ].
Qed.

(* What a candidate says of the canonical simples seen so far. *)
Definition candidate_fits (source : Typing_env.t)
    (seen : list (Index.t * simple)) (possible : Source_env.candidate)
    : Prop :=
  match possible with
  | Source_env.No_simples_in_joined_envs => seen = []
  | Source_env.Canonical_in_all_joined_envs simple =>
    ∀ binding : Index.t * FlambdaModel.simple,
      binding ∈ seen -> snd binding = simple
  | Source_env.Latest_bound_source_var var =>
    Source_env.exists_in_source_env source var = true
  | Source_env.No_canonical_in_source_env => True
  end.

Lemma candidate_step_fits (source : Typing_env.t)
    (seen : list (Index.t * simple))
    (possible possible' : Source_env.candidate)
    (index : Index.t) (canonical : simple) :
  candidate_fits source seen possible ->
  Source_env.candidate_step source possible canonical = Some possible' ->
  candidate_fits source (seen ++ [(index, canonical)]) possible'.
Proof.
  intros Hfits Hstep. unfold Source_env.candidate_step in Hstep.
  assert (Hmaybe :
            candidate_fits source (seen ++ [(index, canonical)])
              match Source_env.source_var_of source canonical with
              | Some var => Source_env.Latest_bound_source_var var
              | None => Source_env.No_canonical_in_source_env
              end).
  { destruct (Source_env.source_var_of source canonical) as [var | ] eqn:Hvar;
      [ | exact I ].
    exact (proj2 (source_var_of_some _ _ _ Hvar)). }
  assert (Hlatest :
            ∀ var : variable,
              Source_env.exists_in_source_env source var = true ->
              match Source_env.source_var_of source canonical with
              | Some var0 =>
                option_map Source_env.Latest_bound_source_var
                  (Source_env.latest_bound source var var0)
              | None => Some (Source_env.Latest_bound_source_var var)
              end
              = Some possible' ->
              candidate_fits source (seen ++ [(index, canonical)]) possible').
  { intros var Hvar Hwith.
    destruct (Source_env.source_var_of source canonical) as [var0 | ]
      eqn:Hvar0.
    - destruct (Source_env.latest_bound source var var0) as [w | ] eqn:Hw;
        [ | discriminate ].
      injection Hwith as <-.
      destruct (latest_bound_cases _ _ _ _ Hw) as [-> | ->];
        [ exact Hvar | exact (proj2 (source_var_of_some _ _ _ Hvar0)) ].
    - injection Hwith as <-. exact Hvar. }
  destruct possible as [ | | shared | var].
  - injection Hstep as <-. cbn in Hfits. subst seen.
    intros binding [<- | []]. reflexivity.
  - injection Hstep as <-. exact Hmaybe.
  - destruct (simple_equal canonical shared) eqn:Hequal.
    + injection Hstep as <-. apply simple_equal_decides_equality in Hequal.
      subst shared. intros binding Hin. apply in_app_or in Hin.
      destruct Hin as [Hin | [<- | []]]; [ exact (Hfits _ Hin) | reflexivity ].
    + destruct (Source_env.source_var_of source shared) as [var | ] eqn:Hvar.
      * exact (Hlatest var (proj2 (source_var_of_some _ _ _ Hvar)) Hstep).
      * injection Hstep as <-. exact Hmaybe.
  - exact (Hlatest var Hfits Hstep).
Qed.

Lemma candidate_fold_fits (source : Typing_env.t) :
  ∀ (rest seen : list (Index.t * simple))
    (possible result : Source_env.candidate),
    candidate_fits source seen possible ->
    List.fold_left
      (fun possible binding =>
         let* possible := possible in
         Source_env.candidate_step source possible (snd binding))
      rest (Some possible)
    = Some result ->
    candidate_fits source (seen ++ rest) result.
Proof.
  induction rest as [ | [index canonical] rest IH];
    intros seen possible result Hfits Hfold.
  - injection Hfold as <-. rewrite app_nil_r. exact Hfits.
  - cbn [List.fold_left snd] in Hfold.
    destruct (Source_env.candidate_step source possible canonical)
      as [possible' | ] eqn:Hstep.
    + pose proof (candidate_step_fits _ _ _ _ index _ Hfits Hstep) as Hfits'.
      pose proof (IH _ _ _ Hfits' Hfold) as Hresult.
      rewrite <- app_assoc in Hresult. exact Hresult.
    + exfalso. clear -Hfold.
      induction rest as [ | b rest IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma equal_in_bindings_true (envs : Joined_envs.t) (simple : simple) :
  ∀ bindings : list (Index.t * FlambdaModel.simple),
    Joined_envs.equal_in_bindings envs simple bindings = Some true ->
    ∀ (index : Index.t) (canonical : FlambdaModel.simple),
      (index, canonical) ∈ bindings ->
      canonical = simple
      ∨ ∃ env : Typing_env.t,
          Joined_envs.get_nth_joined_env envs index = Some env
          ∧ canonical
            = Typing_env.get_canonical_simple_ignoring_name_mode env simple.
Proof.
  induction bindings as [ | [index0 canonical0] bindings IH];
    intros Hequal index canonical Hin; [ destruct Hin | ].
  cbn [Joined_envs.equal_in_bindings] in Hequal.
  destruct (simple_equal canonical0 simple) eqn:Hsame.
  - destruct Hin as [Heq | Hin]; [ | exact (IH Hequal _ _ Hin) ].
    injection Heq as <- <-. left.
    apply simple_equal_decides_equality. exact Hsame.
  - destruct (Joined_envs.get_nth_joined_env envs index0) as [env | ] eqn:Hnth;
      [ | discriminate ].
    destruct (simple_equal canonical0
                (Typing_env.get_canonical_simple_ignoring_name_mode env
                   simple))
      eqn:Hcanonical; [ | discriminate ].
    destruct Hin as [Heq | Hin]; [ | exact (IH Hequal _ _ Hin) ].
    injection Heq as <- <-. right. exists env. split; [ exact Hnth | ].
    apply simple_equal_decides_equality. exact Hcanonical.
Qed.

(* A canonical simple of an environment that has its invariant is in
   scope where the simple is. *)
Lemma canonical_in_scope (tenv : Typing_env.t) (simple : simple) :
  typing_env_invariant tenv ->
  simple_in_scope tenv simple ->
  simple_in_scope tenv
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv simple).
Proof.
  intros [Hwf _] Hscope.
  assert (Hname :
            ∀ n : name,
              (∀ x : variable, n = Name_var x -> knows tenv x) ->
              simple_in_scope tenv
                (Aliases.get_canonical_ignoring_name_mode
                   (Typing_env.aliases tenv) n)).
  { intros n Hn y Hy.
    destruct (Aliases.get_canonical_ignoring_name_mode
                (Typing_env.aliases tenv) n) as [z | sym | c] eqn:Hcanonical;
      [ | destruct Hy | destruct Hy ].
    destruct Hy as [<- | []].
    destruct (canonical_element_mentioned (Typing_env.aliases tenv) n
                (Name_var z) Hcanonical) as [Heq | Hmentioned].
    - apply Hn. symmetry. exact Heq.
    - exact (well_formed_mentioned_known tenv (Name_var z) Hwf Hmentioned). }
  destruct simple as [x | sym | c]; cbn.
  - apply Hname. intros y Heq. injection Heq as <-. apply Hscope.
    left. reflexivity.
  - apply Hname. intros y Heq. discriminate.
  - exact Hscope.
Qed.

(* A variable that is not its own canonical simple is known. *)
Lemma demoted_known (tenv : Typing_env.t) (var : variable) :
  typing_env_invariant tenv ->
  Typing_env.get_canonical_simple_ignoring_name_mode tenv (Var var) <> Var var
  -> knows tenv var.
Proof.
  intros [Hwf _] Hne.
  apply (well_formed_mentioned_known tenv (Name_var var) Hwf).
  left. cbn in Hne. unfold Aliases.get_canonical_ignoring_name_mode in Hne.
  unfold Aliases.canonical in Hne. cbn in Hne. unfold demoted_to.
  destruct (Keyed.lookup name_eqb (Name_var var)
              (Aliases.canonical_elements (Typing_env.aliases tenv)))
    as [canonical | ]; [ | contradiction ].
  exists canonical. reflexivity.
Qed.

(* What a simple of the target is to the vector that it stands for: at
   each index the canonical simple there is that simple, or is the
   canonical simple there of that simple. *)
Definition stands_in_for (s : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (joined : simple) : Prop :=
  ∀ (index : Index.t) (canonical : simple),
    Simples_in_joined_envs.find index canonicals = Some canonical ->
    canonical = joined
    ∨ ∃ env : Typing_env.t,
        Join_env.joined_env s index = Some env
        ∧ canonical
          = Typing_env.get_canonical_simple_ignoring_name_mode env joined.

Lemma canonical_simples_cases (s : Join_env.t)
    (canonicals : Simples_in_joined_envs.t)
    (result : Join_env.canonical_in_target_env) :
  Join_env.n_way_join_canonical_simples s canonicals = Some result ->
  match result with
  | Join_env.Canonical_in_source_env simple =>
    stands_in_for s canonicals simple
    ∧ ∀ x : variable,
        x ∈ free_variables_of_simple simple ->
        Source_env.exists_in_source_env (Join_env.source_env s) x = true
  | Join_env.Import_from_all_joined_envs var =>
    stands_in_for s canonicals (Var var)
  | Join_env.Existential_for_these_canonicals canonicals' =>
    canonicals' = canonicals
  end.
Proof.
  unfold Join_env.n_way_join_canonical_simples. intros H.
  destruct (Source_env.candidate_canonical_in_source_env
              (Join_env.source_env s) canonicals)
    as [candidate | ] eqn:Hcandidate; [ | discriminate ].
  pose proof (candidate_fold_fits (Join_env.source_env s)
                (Keyed.compact Index.equal canonicals) []
                Source_env.No_simples_in_joined_envs candidate eq_refl
                Hcandidate) as Hfits.
  cbn [app] in Hfits.
  destruct candidate as [ | | shared | var].
  - injection H as <-. reflexivity.
  - injection H as <-. reflexivity.
  - assert (Hstands : stands_in_for s canonicals shared).
    { intros index c Hfind. left. apply (Hfits (index, c)).
      apply (keyed_in_compact Index.equal index_equal_decides). exact Hfind. }
    destruct shared as [x | sym | c0].
    + destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
        eqn:Hx; injection H as <-.
      * split; [ exact Hstands | intros y [<- | []]; exact Hx ].
      * exact Hstands.
    + injection H as <-. split; [ exact Hstands | intros y [] ].
    + injection H as <-. split; [ exact Hstands | intros y [] ].
  - destruct (Joined_envs.equal_in_all_joined_envs (Join_env.joined_envs s)
                (Var var) canonicals) as [equal | ] eqn:Hequal;
      [ | discriminate ].
    injection H as <-. destruct equal; [ | reflexivity ].
    split.
    + intros index c Hfind.
      apply (equal_in_bindings_true _ _ _ Hequal).
      apply (keyed_in_compact Index.equal index_equal_decides). exact Hfind.
    + intros y [<- | []]. exact Hfits.
Qed.

Lemma agrees_simple (use : Typing_env.t) (valuation valuation' : env)
    (simple : simple) (v : value) :
  agrees_on_known use valuation valuation' ->
  simple_in_scope use simple ->
  eval_simple valuation simple = Some v ->
  eval_simple valuation' simple = Some v.
Proof.
  intros Hagrees Hscope. destruct simple as [x | sym | c]; cbn;
    [ | auto | auto ].
  rewrite (Hagrees x (Hscope x (or_introl eq_refl))). auto.
Qed.

Lemma stands_upper (s : Join_env.t) (canonicals : Simples_in_joined_envs.t)
    (joined : simple) (index : Index.t) (simple : simple)
    (use : Typing_env.t) (statics : static_env) (valuation valuation' : env)
    (v : value) :
  stands_in_for s canonicals joined ->
  typing_env_invariant use ->
  Join_env.joined_env s index = Some use ->
  simple_in_scope use simple ->
  Simples_in_joined_envs.find index canonicals
  = Some (Typing_env.get_canonical_simple_ignoring_name_mode use simple) ->
  typing_env_satisfied_strictly statics valuation use ->
  agrees_on_known use valuation valuation' ->
  eval_simple valuation simple = Some v ->
  eval_simple valuation' joined = Some v.
Proof.
  intros Hstands Hinvariant Hindex Hscope Hfind Hsatisfied Hagrees Hv.
  pose proof (canonical_in_scope use simple Hinvariant Hscope) as Hscope_c.
  rewrite <- (canonical_ignoring_name_mode_satisfied _ _ _ _ simple Hsatisfied)
    in Hv.
  destruct (Hstands _ _ Hfind) as [Heq | [env [Henv Heq]]].
  - rewrite Heq in Hv, Hscope_c.
    exact (agrees_simple _ _ _ _ _ Hagrees Hscope_c Hv).
  - rewrite Hindex in Henv. injection Henv as <-.
    rewrite Heq in Hv, Hscope_c.
    rewrite (canonical_ignoring_name_mode_satisfied _ _ _ _ joined Hsatisfied)
      in Hv.
    apply (agrees_simple _ _ _ _ _ Hagrees); [ | exact Hv ].
    destruct joined as [x | sym | c0]; [ | intros y [] | intros y [] ].
    intros y [<- | []].
    destruct (simple_equal
                (Typing_env.get_canonical_simple_ignoring_name_mode use
                   (Var x))
                (Var x)) eqn:Hsame.
    + apply simple_equal_decides_equality in Hsame. rewrite Hsame in Hscope_c.
      apply Hscope_c. left. reflexivity.
    + apply (demoted_known use x Hinvariant). intros Habs.
      apply simple_equal_decides_equality in Habs. congruence.
Qed.

Theorem import_type_valid : import_type_valid_statement.
Proof.
  intros p fuel s s' ty Hfresh Hvalid Himport.
  exact (import_type_with_keeps _ _ _ _ (import_var_keeps p fuel Hfresh) Hvalid
           Himport).
Qed.

Theorem n_way_join_simples_valid : n_way_join_simples_valid_statement.
Proof.
  intros p fuel s s' kind simples result Hfresh Hvalid Hwf Hdistinct Hscope
    Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as <- <-.
    split; [ exact Hvalid | split; [ apply grows_refl | ] ].
    split; [ reflexivity | intros joined H; discriminate ]. }
  set (simples := first :: rest) in *. clearbody simples. clear first rest.
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode
              (Join_env.joined_envs s) simples)
    as [canonicals | ] eqn:Hcanonicals; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [canonical | ] eqn:Hcanonical; [ | discriminate ].
  pose proof (canonical_simples_cases _ _ _ Hcanonical) as Hcases.
  assert (Hfound :
            ∀ (s1 : Join_env.t) (index : Index.t) (simple : simple)
              (use : Typing_env.t),
              grows s s1 ->
              In (index, simple) simples ->
              Join_env.joined_env s1 index = Some use ->
              Join_env.joined_env s index = Some use
              ∧ simple_in_scope use simple
              ∧ Simples_in_joined_envs.find index canonicals
                = Some
                    (Typing_env.get_canonical_simple_ignoring_name_mode use
                       simple)).
  { intros s1 index simple use [Henvs _] Hin Huse. rewrite Henvs in Huse.
    split; [ exact Huse | ].
    destruct (Hscope _ _ Hin) as [use1 [Huse1 Hscope1]].
    rewrite Huse in Huse1. injection Huse1 as <-.
    split; [ exact Hscope1 | ].
    destruct (canonical_simples_find _ _ _ _ _ Hcanonicals Hdistinct Hin)
      as [env [Hnth Hfind]].
    unfold Join_env.joined_env in Huse. rewrite Hnth in Huse.
    injection Huse as <-. exact Hfind. }
  assert (Hupper :
            ∀ (joined : simple) (s1 : Join_env.t),
              grows s s1 ->
              stands_in_for s canonicals joined ->
              ∀ (index : Index.t) (simple : simple) (use : Typing_env.t)
                (statics : static_env) (valuation valuation' : env)
                (v : value),
                In (index, simple) simples ->
                Join_env.joined_env s1 index = Some use ->
                typing_env_satisfied_strictly statics valuation use ->
                extends_input Join_env.joined_env
                  Join_env.definitions_of_existentials s1 index valuation
                  valuation' ->
                eval_simple valuation simple = Some v ->
                eval_simple valuation' joined = Some v).
  { intros joined s1 Hgrows Hstands index simple use statics valuation
      valuation' v Hin Huse Hsatisfied [use0 [Huse0 [Hagrees _]]] Hv.
    rewrite Huse in Huse0. injection Huse0 as <-.
    destruct (Hfound _ _ _ _ Hgrows Hin Huse) as [Huse_s [Hscope1 Hfind]].
    exact (stands_upper s canonicals joined index simple use statics valuation
             valuation' v Hstands (Hwf _ _ Huse_s) Huse_s Hscope1 Hfind
             Hsatisfied Hagrees Hv). }
  destruct canonical as [simple0 | var | canonicals'].
  - injection Hjoin as <- <-. destruct Hcases as [Hstands Hsource].
    split; [ exact Hvalid | split; [ apply grows_refl | ] ].
    split; [ discriminate | ].
    intros joined Hjoined. injection Hjoined as <-. split.
    + intros x Hx. rewrite kind_in_target_env_exists.
      unfold Bindings_in_target_env.exists_in_target_env.
      pose proof (Hsource x Hx) as Hexists.
      unfold Join_env.source_env in Hexists. rewrite Hexists. reflexivity.
    + exact (Hupper _ _ (grows_refl s) Hstands).
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as <- <-.
    destruct (import_var_keeps p fuel Hfresh _ _ _ Hvalid Himport)
      as [Hvalid1 [Hgrows1 Hexists1]].
    split; [ exact Hvalid1 | split; [ exact Hgrows1 | ] ].
    split; [ discriminate | ].
    intros joined Hjoined. injection Hjoined as <-. split.
    + intros x [<- | []]. exact Hexists1.
    + exact (Hupper _ _ Hgrows1 Hcases).
  - subst canonicals'.
    destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as <- <-.
    destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
      as [Hvalid1 [Hgrows1 [Hexistential
                              [simples' [kind' [Hdefinition Hequal]]]]]].
    split; [ exact Hvalid1 | split; [ exact Hgrows1 | ] ].
    split; [ discriminate | ].
    intros joined Hjoined. injection Hjoined as <-. split.
    + intros y [<- | []]. rewrite kind_in_target_env_exists.
      unfold Bindings_in_target_env.exists_in_target_env.
      unfold is_existential in Hexistential. rewrite Hexistential.
      rewrite orb_true_r. reflexivity.
    + intros index simple use statics valuation valuation' v Hin Huse
        Hsatisfied [use0 [Huse0 [_ Hholds]]] Hv.
      destruct (Hfound _ _ _ _ Hgrows1 Hin Huse) as [Huse_s [Hscope1 Hfind]].
      rewrite (simples_equal_find _ _ Hequal) in Hfind.
      apply (Hholds x simples' kind' _ v Hdefinition Hfind).
      rewrite (canonical_ignoring_name_mode_satisfied _ _ _ _ simple
                 Hsatisfied).
      exact Hv.
Qed.
Lemma import_name_keeps (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (n : name) :
  fresh_avoids p ->
  state_valid s ->
  Join_env.import_name_transitive p fuel s n = Some s' ->
  state_valid s'
  ∧ grows s s'
  ∧ ∀ x : variable,
      x ∈ free_variables_of_name n -> exists_in_target s' x = true.
Proof.
  intros Hfresh Hvalid Himport. destruct n as [x | sym].
  - destruct (import_var_keeps p fuel Hfresh _ _ _ Hvalid Himport)
      as [Hvalid' [Hgrows' Hexists']].
    split; [ exact Hvalid' | split; [ exact Hgrows' | ] ].
    intros y [<- | []]. exact Hexists'.
  - injection Himport as <-.
    split; [ exact Hvalid | split; [ apply grows_refl | intros y [] ] ].
Qed.

Lemma import_equations_keeps (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p ->
  ∀ (equations : list (name * flambda_type)) (s s' : Join_env.t),
    state_valid s ->
    List.fold_left
      (fun env equation =>
         let* env := env in
         let* env :=
           Join_env.import_name_transitive p fuel env (fst equation)
         in
         Join_env.import_type_transitive p fuel env (snd equation))
      equations (Some s)
    = Some s' ->
    state_valid s'
    ∧ grows s s'
    ∧ ∀ x : variable,
        x ∈ free_variables_of_env_extension (Env_extension equations) ->
        exists_in_target s' x = true.
Proof.
  intros Hfresh.
  induction equations as [ | [n ty] equations IH]; intros s s' Hvalid Hfold.
  - injection Hfold as <-.
    split; [ exact Hvalid | split; [ apply grows_refl | intros x [] ] ].
  - cbn [List.fold_left fst snd] in Hfold.
    destruct (Join_env.import_name_transitive p fuel s n) as [s1 | ]
      eqn:Hname.
    + destruct (import_name_keeps _ _ _ _ _ Hfresh Hvalid Hname)
        as [Hvalid1 [Hgrows1 Hexists1]].
      destruct (Join_env.import_type_transitive p fuel s1 ty) as [s2 | ]
        eqn:Htype.
      * destruct (import_type_valid p fuel s1 s2 ty Hfresh Hvalid1 Htype)
          as [Hvalid2 [Hgrows2 Hexists2]].
        destruct (IH _ _ Hvalid2 Hfold) as [Hvalid' [Hgrows' Hexists']].
        split; [ exact Hvalid' | ].
        split;
          [ exact (grows_trans _ _ _ Hgrows1
                     (grows_trans _ _ _ Hgrows2 Hgrows'))
          | ].
        intros x Hx. cbn [free_variables_of_env_extension flat_map fst snd]
          in Hx.
        apply in_app_or in Hx. destruct Hx as [Hx | Hx];
          [ | exact (Hexists' x Hx) ].
        apply in_app_or in Hx. destruct Hx as [Hx | Hx].
        -- apply (grows_exists _ _ _ (grows_trans _ _ _ Hgrows2 Hgrows')).
           exact (Hexists1 x Hx).
        -- apply (grows_exists _ _ _ Hgrows'). exact (Hexists2 x Hx).
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
    + exfalso. clear -Hfold.
      induction equations as [ | e equations IH];
        [ discriminate | exact (IH Hfold) ].
Qed.

Theorem n_way_join_env_extension_valid
  : n_way_join_env_extension_valid_statement.
Proof.
  intros p fuel s extensions result Hfresh Heqb Hvalid Hscope Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others].
  { injection Hjoin as <-. split; [ reflexivity | ].
    intros joined s' H. discriminate. }
  split; [ destruct (forallb _ others); [ | injection Hjoin as <- ];
           [ | discriminate ];
           destruct (List.fold_left _ _ _); [ | discriminate ];
           injection Hjoin as <-; discriminate | ].
  intros joined s' Hresult.
  destruct (forallb
              (fun binding => Join_env.env_extension_eqb p (snd binding) first)
              others) eqn:Hall.
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as <-. injection Hresult as <- <-.
    destruct first as [equations]. cbn [Join_env.equations_of] in Hfold.
    destruct (import_equations_keeps p fuel Hfresh _ _ _ Hvalid Hfold)
      as [Hvalid1 [Hgrows1 Hexists1]].
    split; [ exact Hvalid1 | split; [ exact Hgrows1 | ] ].
    split; [ exact Hexists1 | ].
    intros index extension use statics valuation valuation' Hin Huse
      Hsatisfied [use0 [Huse0 [Hagrees _]]] Hextension.
    rewrite Huse in Huse0. injection Huse0 as <-.
    destruct Hgrows1 as [Henvs _]. rewrite Henvs in Huse.
    destruct (Hscope _ _ Hin) as [use1 [Huse1 Hscope1]].
    rewrite Huse in Huse1. injection Huse1 as <-.
    assert (Hmoved : env_extension_satisfied statics valuation' extension).
    { apply (env_extension_satisfied_coincide statics valuation valuation'
               extension); [ | exact Hextension ].
      intros x Hx. symmetry. apply Hagrees. apply Hscope1. exact Hx. }
    destruct Hin as [Heq | Hin].
    + injection Heq as _ <-. exact Hmoved.
    + rewrite forallb_forall in Hall. specialize (Hall _ Hin). cbn in Hall.
      exact (Heqb _ _ _ _ Hall Hmoved).
  - injection Hjoin as <-. injection Hresult as <- <-.
    split; [ exact Hvalid | split; [ apply grows_refl | ] ].
    split; [ intros x [] | ].
    intros index extension use statics valuation valuation' _ _ _ _ _.
    apply env_extension_lookup. intros n ty Hlookup. discriminate.
Qed.

(* ======================================================================== *)
(* The state of a join: kinds                                               *)
(* ======================================================================== *)

Definition vector_of_kind (s : Join_env.t)
    (simples : Simples_in_joined_envs.t) (kind : flambda_kind) : Prop :=
  ∀ (index : Index.t) (simple : simple) (use : Typing_env.t),
    Simples_in_joined_envs.find index simples = Some simple ->
    Join_env.joined_env s index = Some use ->
    simple_of_kind use simple kind.

Lemma same_readings_kinded (s s' : Join_env.t) :
  same_readings s s' -> state_kinded s -> state_kinded s'.
Proof.
  destruct s, s'. unfold same_readings. cbn.
  intros [-> [-> [-> ->]]] Hkinded. exact Hkinded.
Qed.

Lemma existential_keeps_kinded (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded s ->
  vector_of_kind s canonicals kind ->
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  state_kinded s'.
Proof.
  intros Hfresh Hvalid Hkinded Hvector Hmake.
  pose proof Hvalid as Hvalid0.
  destruct Hvalid
    as [[Hsource [Huses Hexistentials]]
          [Hnew [[Hrecorded [Hvectors Hdefined_all]] [Himported Himports]]]].
  unfold Join_env.existential_for_these_canonicals,
    Bindings_in_target_env.existential_for_these_simples in Hmake.
  destruct (Bindings_in_target_env.has_existential_for_these_simples
              (Join_env.bindings s) canonicals) as [existing | ] eqn:Hhas.
  - destruct (Hvectors _ _ Hhas) as [simples' [kind' [Hdefinition Hequal]]].
    assert (Hbound : is_bound existing (Join_env.definitions_of_existentials s)
                     = true).
    { apply is_bound_lookup. eexists. exact Hdefinition. }
    rewrite Hbound in Hmake. injection Hmake as <- <-.
    match goal with
    | |- state_kinded ?s' =>
      assert (Hsame : same_readings s s') by (repeat split)
    end.
    exact (same_readings_kinded _ _ Hsame Hkinded).
  - set (x0 :=
           Join_env.fresh p
             (Bindings_in_target_env.names_to_avoid (Join_env.bindings s)))
      in *.
    assert (Hx0 : ~ x0 ∈ names_to_avoid s) by apply Hfresh.
    assert (Hunbound :
              is_bound x0 (Join_env.definitions_of_existentials s) = false).
    { destruct (is_bound x0 (Join_env.definitions_of_existentials s))
        eqn:Hb; [ | reflexivity ].
      exfalso. apply is_bound_lookup in Hb. destruct Hb as [[simples k] Hd].
      apply Hx0. apply Hexistentials. unfold is_existential.
      apply is_bound_lookup. exists k. exact (Hrecorded _ _ _ Hd). }
    clearbody x0. cbv beta iota zeta in Hmake.
    rewrite Hunbound in Hmake. injection Hmake as <- <-.
    destruct Hkinded as [Hsources [Himports_kinded Hdefinitions]].
    split; [ exact Hsources | ]. split; [ exact Himports_kinded | ].
    intros y simples k index simple use Hy Hfind Huse.
    cbn [Join_env.definitions_of_existentials lookup] in Hy.
    destruct (Pos.eqb y x0).
    + injection Hy as <- <-. exact (Hvector _ _ _ Hfind Huse).
    + exact (Hdefinitions _ _ _ _ _ _ Hy Hfind Huse).
Qed.

Lemma get_alias_in_aliases (ty : flambda_type) (alias : simple) :
  get_alias_opt ty = Some alias -> (alias, kind ty) ∈ aliases_with_kinds ty.
Proof.
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; cbn; try discriminate;
    intros H; injection H as <-; left; reflexivity.
Qed.

Lemma get_alias_alias_type_of (kind : flambda_kind) (simple : simple) :
  get_alias_opt (alias_type_of kind simple) = Some simple.
Proof. destruct kind as [ | k | | ]; [ | destruct k | | ]; reflexivity. Qed.

Lemma types_of_imported_var_spec
    (envs :
       list (Index.t * (Typing_env.t * Keyed.finite_map name flambda_type)))
    (var : variable) (kind : flambda_kind) :
  ∀ types : Keyed.finite_map Index.t flambda_type,
    Joined_envs.types_of_imported_var envs var kind = Some types ->
    (∀ (index : Index.t) (env : Typing_env.t)
       (equations : Keyed.finite_map name flambda_type),
       (index, (env, equations)) ∈ envs ->
       knows env var ->
       ∃ ty : flambda_type,
         Typing_env.find env (Name_var var) (Some kind) = Some ty)
    ∧ ∀ (index : Index.t) (ty : flambda_type),
        (index, ty) ∈ types ->
        ∃ (env : Typing_env.t)
          (equations : Keyed.finite_map name flambda_type),
          (index, (env, equations)) ∈ envs
          ∧ Typing_env.find env (Name_var var) (Some kind) = Some ty.
Proof.
  induction envs as [ | [index0 [env0 equations0]] envs IH]; intros types H.
  - injection H as <-. split; [ intros index env equations [] | ].
    intros index ty [].
  - cbn [Joined_envs.types_of_imported_var] in H.
    destruct (Joined_envs.types_of_imported_var envs var kind)
      as [types0 | ]; [ | discriminate ].
    destruct (IH _ eq_refl) as [Hknown Hfound].
    destruct (Typing_env.mem None env0 (Name_var var)) eqn:Hmem.
    + destruct (Typing_env.find env0 (Name_var var) (Some kind)) as [ty0 | ]
        eqn:Hfind; [ | discriminate ].
      injection H as <-. split.
      * intros index env equations [Heq | Hin] Hk.
        -- injection Heq as <- <- <-. exists ty0. exact Hfind.
        -- exact (Hknown _ _ _ Hin Hk).
      * intros index ty [Heq | Hin].
        -- injection Heq as <- <-. exists env0, equations0.
           split; [ left; reflexivity | exact Hfind ].
        -- destruct (Hfound _ _ Hin) as [env [equations [Hin' Hfind']]].
           exists env, equations. split; [ right; exact Hin' | exact Hfind' ].
    + injection H as <-. split.
      * intros index env equations [Heq | Hin] Hk.
        -- injection Heq as <- <- <-. unfold knows in Hk. congruence.
        -- exact (Hknown _ _ _ Hin Hk).
      * intros index ty Hin.
        destruct (Hfound _ _ Hin) as [env [equations [Hin' Hfind']]].
        exists env, equations. split; [ right; exact Hin' | exact Hfind' ].
Qed.

Lemma canonicals_of_alias_types_spec (envs : Joined_envs.t) :
  ∀ (types : list (Index.t * flambda_type))
    (canonicals : Simples_in_joined_envs.t),
    Joined_envs.canonicals_of_alias_types envs types = Some (Some canonicals)
    ->
    ∀ (index : Index.t) (canonical : simple),
      (index, canonical) ∈ canonicals ->
      ∃ (ty : flambda_type) (env : Typing_env.t) (alias : simple),
        (index, ty) ∈ types
        ∧ Joined_envs.get_nth_joined_env envs index = Some env
        ∧ get_alias_opt ty = Some alias
        ∧ canonical
          = Typing_env.get_canonical_simple_ignoring_name_mode env alias.
Proof.
  induction types as [ | [index0 ty0] types IH]; intros canonicals H.
  - injection H as <-. intros index canonical [].
  - cbn [Joined_envs.canonicals_of_alias_types] in H.
    destruct (Joined_envs.get_nth_joined_env envs index0) as [env0 | ]
      eqn:Hnth; [ | discriminate ].
    destruct (Joined_envs.canonicals_of_alias_types envs types)
      as [[canonicals0 | ] | ]; try discriminate.
    + destruct (get_alias_opt ty0) as [alias0 | ] eqn:Halias;
        [ | discriminate ].
      injection H as <-. intros index canonical [Heq | Hin].
      * injection Heq as <- <-. exists ty0, env0, alias0.
        split; [ left; reflexivity | ].
        split; [ exact Hnth | split; [ exact Halias | reflexivity ] ].
      * destruct (IH _ eq_refl _ _ Hin) as [ty [env [alias [Hin' Hrest]]]].
        exists ty, env, alias. split; [ right; exact Hin' | exact Hrest ].
    + destruct (get_alias_opt ty0); discriminate.
Qed.

Section Kinds.

Abbreviation env_kinded := typing_env_kinded.

Lemma canonical_kinded (tenv : Typing_env.t) (s : simple)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  typing_env_kinded tenv ->
  simple_of_kind tenv s kind ->
  simple_of_kind tenv
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv s) kind.
Proof.
  intros [Hwf _] [_ [Haliases _]] Hkind.
  exact (canonical_ignoring_name_mode_kinded tenv s kind Hwf Haliases Hkind).
Qed.

Lemma symbol_kinded (tenv : Typing_env.t) (s : symbol) (kind : flambda_kind) :
  typing_env_kinded tenv -> simple_of_kind tenv (Symbol s) kind ->
  kind = Value.
Proof.
  intros [_ [_ Hsymbols]] Hkind.
  exact (symbol_of_kind_value tenv s kind Hsymbols Hkind).
Qed.

(* A symbol that is of a kind in an environment that has its
   invariant is known to it: one that has a type is defined. *)
Lemma symbol_of_kind_known (tenv : Typing_env.t) (s : symbol)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  simple_of_kind tenv (Symbol s) kind ->
  Typing_env.mem None tenv (Name_symbol s) = true.
Proof.
  intros [Hwf _] Hkind. apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [[_ [_ [_ [_ [Hsymbols _]]]]] _].
  change (Symbol s) with (simple_of_name (Name_symbol s)) in Hkind.
  apply simple_of_kind_name in Hkind.
  destruct Hkind as [[ty [btm [Hlookup _]]] | [_ [Hmem _]]];
    [ exact (Hsymbols s ty btm Hlookup) | exact Hmem ].
Qed.

Lemma find_kinded_in (tenv : Typing_env.t) (n : name)
    (kind : option flambda_kind) (ty : flambda_type) :
  typing_env_kinded tenv -> Typing_env.find tenv n kind = Some ty ->
  type_kinded tenv ty.
Proof.
  intros [Htypes _] Hfind. exact (find_kinded tenv n kind ty Htypes Hfind).
Qed.

Abbreviation envs_invariant := (joined_envs_are typing_env_invariant).

Abbreviation envs_kinded := (joined_envs_are env_kinded).

Lemma grows_envs_invariant (s s' : Join_env.t) :
  grows s s' -> envs_invariant s -> envs_invariant s'.
Proof.
  intros [Henvs _] H index use Huse. rewrite Henvs in Huse.
  exact (H _ _ Huse).
Qed.

Lemma grows_envs_kinded (s s' : Join_env.t) :
  grows s s' -> envs_kinded s -> envs_kinded s'.
Proof.
  intros [Henvs _] H index use Huse. rewrite Henvs in Huse.
  exact (H _ _ Huse).
Qed.

(* What is shown of a function that imports a variable, for kinds. *)
Definition import_keeps_kinded
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    state_valid s ->
    state_kinded s ->
    envs_invariant s ->
    envs_kinded s ->
    import_var s x = Some s' ->
    state_kinded s'.

Lemma import_vars_keeps_kinded
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps import_var ->
  import_keeps_kinded import_var ->
  ∀ s s' : Join_env.t,
    state_valid s ->
    state_kinded s ->
    envs_invariant s ->
    envs_kinded s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    state_kinded s'.
Proof.
  intros Himport Himport_kinded.
  induction vars as [ | var vars IH];
    intros s s' Hvalid Hkinded Hinvariant Henvs Hfold.
  - injection Hfold as <-. exact Hkinded.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Himport _ _ _ Hvalid Hvar) as [Hvalid1 [Hgrows1 _]].
      apply (IH s1 s' Hvalid1); try assumption.
      * exact (Himport_kinded _ _ _ Hvalid Hkinded Hinvariant Henvs Hvar).
      * exact (grows_envs_invariant _ _ Hgrows1 Hinvariant).
      * exact (grows_envs_kinded _ _ Hgrows1 Henvs).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_alias_keeps_kinded (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded s ->
  vector_of_kind s canonicals kind ->
  state_kinded (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hfresh Hvalid Hkinded Hvector.
  unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  pose proof (existential_keeps_kinded _ _ _ _ _ _ Hfresh Hvalid Hkinded
                Hvector Hmake) as Hkinded1.
  match goal with
  | |- state_kinded ?s' =>
    assert (Hsame : same_readings s1 s') by (repeat split)
  end.
  exact (same_readings_kinded _ _ Hsame Hkinded1).
Qed.

Lemma add_equation_keeps_kinded (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  state_kinded s ->
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  state_kinded s'.
Proof.
  intros Hkinded Hadd.
  exact (same_readings_kinded _ _ (add_equation_same_readings _ _ _ _ Hadd)
           Hkinded).
Qed.

Lemma add_equals_keeps_kinded (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps_kinded import_var ->
  state_valid s ->
  state_kinded s ->
  envs_invariant s ->
  envs_kinded s ->
  vector_of_kind s simples kind ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  state_kinded s'.
Proof.
  intros Hfresh Himport Himport_kinded Hvalid Hkinded Hinvariant Henvs
    Hvector Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ] eqn:Hcanonical;
    [ | | | discriminate ].
  - exact (add_equation_keeps_kinded _ _ _ _ Hkinded Hadd).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    apply (add_equation_keeps_kinded _ _ _ _
             (Himport_kinded _ _ _ Hvalid Hkinded Hinvariant Henvs Hvar) Hadd).
  - injection Hadd as <-.
    pose proof (canonical_simples_cases _ _ _ Hcanonical) as Hcases.
    cbn in Hcases. subst canonicals.
    exact (add_alias_keeps_kinded _ _ _ _ _ Hfresh Hvalid Hkinded Hvector).
Qed.

Lemma concrete_keeps_kinded (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps import_var ->
  import_keeps_kinded import_var ->
  state_valid s ->
  state_kinded s ->
  envs_invariant s ->
  envs_kinded s ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  state_kinded s'.
Proof.
  intros Himport Himport_kinded Hvalid Hkinded Hinvariant Henvs Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    pose proof (add_equation_same_readings _ _ _ _ Hadd) as Hsame.
    apply (import_vars_keeps_kinded import_var (free_variables any_ty) Himport
             Himport_kinded s1 s').
    + exact (same_readings_valid _ _ Hsame Hvalid).
    + exact (same_readings_kinded _ _ Hsame Hkinded).
    + exact (grows_envs_invariant _ _ (same_readings_grows _ _ Hsame)
               Hinvariant).
    + exact (grows_envs_kinded _ _ (same_readings_grows _ _ Hsame) Henvs).
    + exact Hconcrete.
  - injection Hconcrete as <-.
    exact (same_readings_kinded _ _ (add_equations_same_readings s var types)
             Hkinded).
Qed.

Lemma with_import_keeps_kinded (s : Join_env.t) (x : variable)
    (kind : flambda_kind) :
  (∀ (index : Index.t) (use : Typing_env.t),
     Join_env.joined_env s index = Some use ->
     knows use x ->
     simple_of_kind use (Var x) kind) ->
  state_kinded s ->
  state_kinded (with_import s x kind).
Proof.
  intros Hx [Hsources [Himports Hdefinitions]].
  unfold with_import, Bindings_in_target_env.import_from_all_envs.
  destruct (is_bound x
              (Bindings_in_target_env.imported_variables
                 (Join_env.bindings s))).
  - split; [ exact Hsources | split; [ exact Himports | exact Hdefinitions ] ].
  - split; [ exact Hsources | split; [ | exact Hdefinitions ] ].
    intros y k index use Hy Huse Hknown.
    cbn [Join_env.bindings Bindings_in_target_env.imported_variables lookup]
      in Hy.
    destruct (Pos.eqb y x) eqn:Heq.
    + apply Pos.eqb_eq in Heq. subst y. injection Hy as <-.
      exact (Hx _ _ Huse Hknown).
    + exact (Himports _ _ _ _ Hy Huse Hknown).
Qed.

Lemma joined_env_in_compact (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) :
  Join_env.joined_env s index = Some use ->
  ∃ equations : Keyed.finite_map name flambda_type,
    (index, (use, equations))
    ∈ Keyed.compact Index.equal (Join_env.joined_envs s).
Proof.
  unfold Join_env.joined_env, Joined_envs.get_nth_joined_env. intros Huse.
  destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s))
    as [[use0 equations] | ] eqn:Hlookup; [ | discriminate ].
  injection Huse as <-. exists equations.
  apply (keyed_in_compact Index.equal index_equal_decides). exact Hlookup.
Qed.

Lemma in_compact_joined_env (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (equations : Keyed.finite_map name flambda_type) :
  (index, (use, equations))
  ∈ Keyed.compact Index.equal (Join_env.joined_envs s) ->
  Join_env.joined_env s index = Some use.
Proof.
  intros Hin. apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
  unfold Join_env.joined_env, Joined_envs.get_nth_joined_env.
  rewrite Hin. reflexivity.
Qed.

Lemma import_body_keeps_kinded (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  fresh_avoids p ->
  import_keeps import_var ->
  import_keeps_kinded import_var ->
  import_keeps_kinded (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hfresh Himport Himport_kinded s s' x Hvalid Hkinded Hinvariant Henvs
    Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource; [ injection Hbody as <-; exact Hkinded | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; exact Hkinded | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ] eqn:Hkind; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hunknown : ~ knows (Join_env.source_env s) x).
  { unfold knows. unfold Source_env.exists_in_source_env in Hsource.
    rewrite Hsource. discriminate. }
  destruct (with_import_keeps s x kind Hvalid Hunknown
              (kind_of_imported_var_known _ _ _ Hkind))
    as [Hvalid1 [Hgrows1 _]].
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ] eqn:Htypes;
    [ | discriminate ].
  destruct (types_of_imported_var_spec _ _ _ _ Htypes) as [Hknown Hfound].
  assert (Hx :
            ∀ (index : Index.t) (use : Typing_env.t),
              Join_env.joined_env s index = Some use ->
              knows use x ->
              simple_of_kind use (Var x) kind).
  { intros index use Huse Hk.
    destruct (joined_env_in_compact _ _ _ Huse) as [equations Hin].
    exact (Hknown _ _ _ Hin Hk). }
  pose proof (with_import_keeps_kinded s x kind Hx Hkinded) as Hkinded1.
  pose proof (grows_envs_invariant _ _ Hgrows1 Hinvariant) as Hinvariant1.
  pose proof (grows_envs_kinded _ _ Hgrows1 Henvs) as Henvs1.
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ] eqn:Hcanonicals;
      try discriminate.
    apply (add_equals_keeps_kinded p import_var (with_import s x kind) s'
             kind x canonical_simples Hfresh Himport Himport_kinded Hvalid1
             Hkinded1 Hinvariant1 Henvs1);
      [ | exact Hbody ].
    intros index c use Hfind Huse.
    apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
    destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hfind)
      as [ty' [env [alias [Hin' [Hnth [Halias ->]]]]]].
    change (Join_env.joined_env s index = Some use) in Huse.
    change (Join_env.joined_env s index = Some env) in Hnth.
    rewrite Huse in Hnth. injection Hnth as <-.
    apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
    rewrite (keyed_lookup_map_values Index.t Index.equal) in Hin'.
    destruct (Keyed.lookup Index.equal index types) as [ty0 | ] eqn:Hty0;
      [ | discriminate ].
    injection Hin' as Hty'.
    apply (keyed_lookup_in Index.equal index_equal_decides) in Hty0.
    destruct (Hfound _ _ Hty0) as [env0 [equations0 [Hin0 Hfind0]]].
    pose proof (in_compact_joined_env _ _ _ _ Hin0) as Huse0.
    change (Join_env.joined_env s index = Some env0) in Huse0.
    rewrite Huse in Huse0. injection Huse0 as <-.
    apply (canonical_kinded use alias kind (Hinvariant _ _ Huse)
             (Henvs _ _ Huse)).
    destruct (get_alias_opt ty0) as [alias0 | ] eqn:Halias0.
    + subst ty'. rewrite Halias0 in Halias. injection Halias as <-.
      pose proof (get_alias_in_aliases _ _ Halias0) as Hmember.
      rewrite (find_kind _ _ _ _ Hfind0) in Hmember.
      exact (find_kinded_in _ _ _ _ (Henvs _ _ Huse) Hfind0 _ _ Hmember).
    + subst ty'. rewrite get_alias_alias_type_of in Halias.
      injection Halias as <-. exists ty0. exact Hfind0.
  - exact (concrete_keeps_kinded _ _ _ _ _ _ Himport Himport_kinded Hvalid1
             Hkinded1 Hinvariant1 Henvs1 Hbody).
Qed.

Lemma import_var_keeps_kinded (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p ->
  import_keeps_kinded (Join_env.import_var_transitive p fuel).
Proof.
  intros Hfresh. induction fuel as [ | fuel IH].
  - intros s s' x _ _ _ _ H. discriminate.
  - exact (import_body_keeps_kinded p _ Hfresh (import_var_keeps p fuel Hfresh)
             IH).
Qed.


(* ---------------------------------------------------------------------- *)
(* The kind of a variable of the target, from a joined environment         *)
(* ---------------------------------------------------------------------- *)

Lemma simple_of_kind_var_known (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) :
  simple_of_kind tenv (Var x) kind -> knows tenv x.
Proof.
  intros [ty Hfind]. unfold knows, Typing_env.mem.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode' in Hfind.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv))
    as [[ty0 time] | ]; [ reflexivity | discriminate ].
Qed.

Lemma kind_in_target_of_use (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (x : variable) (kind : flambda_kind) :
  state_valid s ->
  state_kinded s ->
  Join_env.joined_env s index = Some use ->
  simple_of_kind use (Var x) kind ->
  exists_in_target s x = true ->
  kind_in_target_env s x = Some kind.
Proof.
  intros [_ [Hnew _]] [Hsources [Himports _]] Huse Hkind Hexists.
  unfold exists_in_target_env in Hexists. unfold kind_in_target_env in *.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types
                 (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty time] | ] eqn:Hsource.
  - rewrite (Hsources _ _ _ _ _ _ Huse Hkind Hsource). reflexivity.
  - destruct (lookup x
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))) as [k | ] eqn:Hexistential.
    + exfalso.
      assert (Hx : is_existential s x).
      { unfold is_existential. apply is_bound_lookup. exists k.
        exact Hexistential. }
      exact (proj2 (Hnew x Hx) _ _ Huse
               (simple_of_kind_var_known _ _ _ Hkind)).
    + destruct (lookup x
                  (Bindings_in_target_env.imported_variables
                     (Join_env.bindings s))) as [k | ] eqn:Himported;
        [ | discriminate ].
      pose proof (Himports _ _ _ _ Himported Huse
                    (simple_of_kind_var_known _ _ _ Hkind)) as Hk.
      rewrite (simple_of_kind_unique _ _ _ _ Hk Hkind). reflexivity.
Qed.

(* A simple that is of a kind in a joined environment, and whose
   variables are in the target, is of that kind in the target. *)
Lemma simple_kind_in_target_of_use (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (simple : simple) (kind : flambda_kind) :
  state_valid s ->
  state_kinded s ->
  envs_invariant s ->
  envs_kinded s ->
  Join_env.joined_env s index = Some use ->
  simple_of_kind use simple kind ->
  simple_in_target kind_in_target_env s simple ->
  simple_of_kind_in_target Join_env.joined_env kind_in_target_env s simple
    kind.
Proof.
  intros Hvalid Hkinded Hinvariant Henvs Huse Hkind Hin.
  destruct simple as [x | sym | c].
  - apply (kind_in_target_of_use s index use x kind Hvalid Hkinded Huse Hkind).
    apply Hin. left. reflexivity.
  - split; [ exact (symbol_kinded _ _ _ (Henvs _ _ Huse) Hkind) | ].
    exists index, use.
    split; [ exact Huse | ].
    exact (symbol_of_kind_known use sym kind (Hinvariant _ _ Huse) Hkind).
  - exact Hkind.
Qed.


(* ---------------------------------------------------------------------- *)
(* The join of simples: kinds                                              *)
(* ---------------------------------------------------------------------- *)

(* A name that is bound is of some kind. *)
Lemma bound_has_kind (tenv : Typing_env.t) (n : name) :
  Typing_env.mem None tenv n = true ->
  ∃ kind : flambda_kind, simple_of_kind tenv (simple_of_name n) kind.
Proof.
  intros Hmem.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty time] | ] eqn:Hlookup.
  - exists (kind ty). apply simple_of_kind_name. left. exists ty, time.
    split; [ exact Hlookup | reflexivity ].
  - exists Value. apply simple_of_kind_name. right.
    split; [ exact Hlookup | split; [ exact Hmem | reflexivity ] ].
Qed.

(* A name that is not its own canonical simple is bound. *)
Lemma demoted_name_known (tenv : Typing_env.t) (n : name) :
  typing_env_invariant tenv ->
  Typing_env.get_canonical_simple_ignoring_name_mode tenv (simple_of_name n)
  <> simple_of_name n ->
  Typing_env.mem None tenv n = true.
Proof.
  intros [Hwf _] Hne.
  apply (well_formed_mentioned_known tenv n Hwf).
  left. unfold demoted_to.
  assert (Hcanonical :
            Aliases.get_canonical_ignoring_name_mode (Typing_env.aliases tenv)
              n
            <> simple_of_name n).
  { destruct n as [x | sym]; exact Hne. }
  unfold Aliases.get_canonical_ignoring_name_mode, Aliases.canonical
    in Hcanonical.
  assert (Hname : name_of_simple (simple_of_name n) = Some n).
  { destruct n; reflexivity. }
  rewrite Hname in Hcanonical.
  destruct (Keyed.lookup name_eqb n
              (Aliases.canonical_elements (Typing_env.aliases tenv)))
    as [canonical | ]; [ | contradiction ].
  exists canonical. reflexivity.
Qed.

(* A simple whose canonical simple is of a kind is of that kind. *)
Lemma kind_of_canonical (tenv : Typing_env.t) (simple : simple)
    (kind : flambda_kind) :
  typing_env_invariant tenv ->
  env_kinded tenv ->
  simple_of_kind tenv
    (Typing_env.get_canonical_simple_ignoring_name_mode tenv simple) kind ->
  simple_of_kind tenv simple kind.
Proof.
  intros Hinvariant Henv Hkind.
  destruct (simple_equal
              (Typing_env.get_canonical_simple_ignoring_name_mode tenv simple)
              simple) eqn:Hsame.
  - apply simple_equal_decides_equality in Hsame. rewrite Hsame in Hkind.
    exact Hkind.
  - assert (Hne :
              Typing_env.get_canonical_simple_ignoring_name_mode tenv simple
              <> simple).
    { intros Habs. apply simple_equal_decides_equality in Habs. congruence. }
    assert (Hbound : ∃ n : name, simple = simple_of_name n).
    { destruct simple as [x | sym | c];
        [ exists (Name_var x); reflexivity
        | exists (Name_symbol sym); reflexivity
        | exfalso; apply Hne; reflexivity ]. }
    destruct Hbound as [n ->].
    destruct (bound_has_kind tenv n (demoted_name_known tenv n Hinvariant Hne))
      as [kind' Hkind'].
    pose proof (canonical_kinded _ _ _ Hinvariant Henv Hkind') as Hcanonical.
    rewrite (simple_of_kind_unique _ _ _ _ Hkind Hcanonical). exact Hkind'.
Qed.

Lemma canonical_simples_binding (envs : Joined_envs.t)
    (simples : list (Index.t * simple)) (canonicals : Simples_in_joined_envs.t)
    (index : Index.t) (canonical : simple) :
  Joined_envs.get_canonical_simples_ignoring_name_mode envs simples
  = Some canonicals ->
  Simples_in_joined_envs.find index canonicals = Some canonical ->
  ∃ simple : FlambdaModel.simple, (index, simple) ∈ simples.
Proof.
  intros Hcanonicals Hfind.
  destruct (canonical_simples_fold envs _ _ _ Hcanonicals) as [Hkeys _].
  apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
  assert (Hin : index ∈ map fst canonicals).
  { apply in_map_iff. exists (index, canonical).
    split; [ reflexivity | exact Hfind ]. }
  rewrite Hkeys, app_nil_r in Hin. apply in_rev in Hin.
  apply in_map_iff in Hin. destruct Hin as [[index' simple] [Heq Hin]].
  cbn in Heq. subst index'. exists simple. exact Hin.
Qed.

Lemma n_way_join_simples_kinds_in :
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    fresh_avoids p ->
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    join_args_distinct simples ->
    join_args_in_scope Join_env.joined_env
      (fun use simple => simple_of_kind use simple kind) s simples ->
    Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
    (∀ sym : symbol, result = Or_bottom.Ok (Symbol sym) -> kind = Value)
    ∧ (state_kinded s ->
       state_kinded s'
       ∧ ∀ joined : simple,
           result = Or_bottom.Ok joined ->
           simple_of_kind_in_target Join_env.joined_env kind_in_target_env
             s' joined kind).
Proof.
  intros p fuel s s' kind simples result Hfresh Hvalid Hinvariant Henvs
    Hdistinct Hscope Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | [index0 simple0] rest].
  { injection Hjoin as <- <-. split; [ intros sym H; discriminate | ].
    intros Hkinded. split; [ exact Hkinded | intros joined H; discriminate ]. }
  set (simples := (index0, simple0) :: rest) in *.
  assert (Hin0 : (index0, simple0) ∈ simples) by (left; reflexivity).
  clearbody simples.
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode
              (Join_env.joined_envs s) simples)
    as [canonicals | ] eqn:Hcanonicals; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [canonical | ] eqn:Hcanonical; [ | discriminate ].
  pose proof (canonical_simples_cases _ _ _ Hcanonical) as Hcases.
  (* Each simple given, its environment and its canonical simple. *)
  assert (Hgiven :
            ∀ (index : Index.t) (simple : simple),
              (index, simple) ∈ simples ->
              ∃ use : Typing_env.t,
                Join_env.joined_env s index = Some use
                ∧ Simples_in_joined_envs.find index canonicals
                  = Some
                      (Typing_env.get_canonical_simple_ignoring_name_mode use
                         simple)
                ∧ simple_of_kind use
                    (Typing_env.get_canonical_simple_ignoring_name_mode use
                       simple)
                    kind).
  { intros index simple Hin.
    destruct (Hscope _ _ Hin) as [use [Huse Hkind]].
    destruct (canonical_simples_find _ _ _ _ _ Hcanonicals Hdistinct Hin)
      as [env [Hnth Hfind]].
    unfold Join_env.joined_env in Huse. rewrite Hnth in Huse.
    injection Huse as <-. exists env. split; [ exact Hnth | ].
    split; [ exact Hfind | ].
    exact (canonical_kinded _ _ _ (Hinvariant _ _ Hnth) (Henvs _ _ Hnth)
             Hkind). }
  assert (Hvector : vector_of_kind s canonicals kind).
  { intros index c use Hfind Huse.
    destruct (canonical_simples_binding _ _ _ _ _ Hcanonicals Hfind)
      as [simple Hin].
    destruct (Hgiven _ _ Hin) as [use' [Huse' [Hfind' Hkind']]].
    rewrite Huse in Huse'. injection Huse' as <-.
    rewrite Hfind in Hfind'. injection Hfind' as ->. exact Hkind'. }
  destruct (Hgiven _ _ Hin0) as [use0 [Huse0 [Hfind0 Hkind0]]].
  (* What stands for the vector is of the kind, in the first
     environment. *)
  assert (Hstands_kind :
            ∀ joined : simple,
              stands_in_for s canonicals joined ->
              simple_of_kind use0 joined kind).
  { intros joined Hstands.
    destruct (Hstands _ _ Hfind0) as [Heq | [env [Henv Heq]]].
    - rewrite <- Heq. exact Hkind0.
    - rewrite Huse0 in Henv. injection Henv as <-.
      apply (kind_of_canonical use0 joined kind (Hinvariant _ _ Huse0)
               (Henvs _ _ Huse0)).
      rewrite <- Heq. exact Hkind0. }
  destruct canonical as [joined0 | var | canonicals'].
  - injection Hjoin as <- <-. destruct Hcases as [Hstands Hsource].
    pose proof (Hstands_kind _ Hstands) as Hkind.
    split.
    + intros sym Heq. injection Heq as ->.
      exact (symbol_kinded _ _ _ (Henvs _ _ Huse0) Hkind).
    + intros Hkinded. split; [ exact Hkinded | ].
      intros joined Heq. injection Heq as <-.
      apply (simple_kind_in_target_of_use s index0 use0 joined0 kind Hvalid
               Hkinded Hinvariant Henvs Huse0 Hkind).
      intros x Hx. rewrite kind_in_target_env_exists.
      unfold Bindings_in_target_env.exists_in_target_env.
      pose proof (Hsource x Hx) as Hexists.
      unfold Join_env.source_env in Hexists. rewrite Hexists. reflexivity.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as <- <-.
    split; [ intros sym H; discriminate | ].
    intros Hkinded.
    destruct (import_var_keeps p fuel Hfresh _ _ _ Hvalid Himport)
      as [Hvalid1 [Hgrows1 Hexists1]].
    pose proof (import_var_keeps_kinded p fuel Hfresh _ _ _ Hvalid Hkinded
                  Hinvariant Henvs Himport) as Hkinded1.
    split; [ exact Hkinded1 | ].
    intros joined Heq. injection Heq as <-.
    assert (Huse1 : Join_env.joined_env s1 index0 = Some use0).
    { destruct Hgrows1 as [Henvs1 _]. rewrite Henvs1. exact Huse0. }
    exact (kind_in_target_of_use s1 index0 use0 var kind Hvalid1 Hkinded1 Huse1
             (Hstands_kind _ Hcases) Hexists1).
  - subst canonicals'.
    destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as <- <-.
    split; [ intros sym H; discriminate | ].
    intros Hkinded.
    destruct (existential_keeps _ _ _ _ _ _ Hfresh Hvalid Hmake)
      as [Hvalid1 [Hgrows1 [Hexistential
                              [simples' [kind' [Hdefinition Hequal]]]]]].
    pose proof (existential_keeps_kinded _ _ _ _ _ _ Hfresh Hvalid Hkinded
                  Hvector Hmake) as Hkinded1.
    split; [ exact Hkinded1 | ].
    intros joined Heq. injection Heq as <-.
    assert (Huse1 : Join_env.joined_env s1 index0 = Some use0).
    { destruct Hgrows1 as [Henvs1 _]. rewrite Henvs1. exact Huse0. }
    destruct Hkinded1 as [_ [_ Hdefinitions1]].
    rewrite (simples_equal_find _ _ Hequal) in Hfind0.
    pose proof (Hdefinitions1 _ _ _ _ _ _ Hdefinition Hfind0 Huse1) as Hkind'.
    rewrite (simple_of_kind_unique _ _ _ _ Hkind' Hkind0) in Hdefinition.
    destruct Hvalid1 as [_ [Hnew1 [[Hrecorded1 _] _]]].
    cbn. unfold kind_in_target_env.
    destruct (Hnew1 x Hexistential) as [Hunknown _].
    unfold knows, Typing_env.mem, Join_env.source_env in Hunknown.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types
                   (Bindings_in_target_env.source_env (Join_env.bindings s1))))
      as [[ty time] | ]; [ exfalso; apply Hunknown; reflexivity | ].
    rewrite (Hrecorded1 _ _ _ Hdefinition). reflexivity.
Qed.


(* ---------------------------------------------------------------------- *)
(* Importing a type, and the join of extensions: kinds                     *)
(* ---------------------------------------------------------------------- *)

(* What is supposed of the aliases of a type: they are made of its
   free variables. *)
Hypothesis aliases_free :
  ∀ (ty : flambda_type) (simple : simple) (kind : flambda_kind) (x : variable),
    (simple, kind) ∈ aliases_with_kinds ty ->
    x ∈ free_variables_of_simple simple ->
    x ∈ free_variables ty.

Hypothesis aliases_free_extension :
  ∀ (extension : env_extension) (simple : simple) (kind : flambda_kind)
    (x : variable),
    (simple, kind) ∈ aliases_of_env_extension extension ->
    x ∈ free_variables_of_simple simple ->
    x ∈ free_variables_of_env_extension extension.

Lemma import_type_kinds_in :
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (ty : flambda_type),
    fresh_avoids p ->
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    Join_env.import_type p fuel s ty = Some s' ->
    state_kinded s ->
    (∃ (index : Index.t) (use : Typing_env.t),
       Join_env.joined_env s index = Some use ∧ type_kinded use ty) ->
    state_kinded s'
    ∧ type_kinded_in_target Join_env.joined_env kind_in_target_env s' ty.
Proof.
  intros p fuel s s' ty Hfresh Hvalid Hinvariant Henvs Himport Hkinded
    [index [use [Huse Hty]]].
  destruct (import_type_valid p fuel s s' ty Hfresh Hvalid Himport)
    as [Hvalid' [Hgrows' Htarget]].
  assert (Hkinded' : state_kinded s').
  { exact (import_vars_keeps_kinded _ (free_variables ty)
             (import_var_keeps p fuel Hfresh)
             (import_var_keeps_kinded p fuel Hfresh) s s' Hvalid Hkinded
             Hinvariant Henvs Himport). }
  split; [ exact Hkinded' | ].
  intros simple kind Hmember.
  assert (Huse' : Join_env.joined_env s' index = Some use).
  { destruct Hgrows' as [Henvs' _]. rewrite Henvs'. exact Huse. }
  apply (simple_kind_in_target_of_use s' index use simple kind Hvalid'
           Hkinded' (grows_envs_invariant _ _ Hgrows' Hinvariant)
           (grows_envs_kinded _ _ Hgrows' Henvs) Huse' (Hty _ _ Hmember)).
  intros x Hx. apply Htarget. exact (aliases_free _ _ _ _ Hmember Hx).
Qed.

Lemma import_name_keeps_kinded (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (n : name) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded s ->
  envs_invariant s ->
  envs_kinded s ->
  Join_env.import_name_transitive p fuel s n = Some s' ->
  state_kinded s'.
Proof.
  intros Hfresh Hvalid Hkinded Hinvariant Henvs Himport.
  destruct n as [x | sym].
  - exact (import_var_keeps_kinded p fuel Hfresh _ _ _ Hvalid Hkinded
             Hinvariant Henvs Himport).
  - injection Himport as <-. exact Hkinded.
Qed.

Lemma import_equations_keeps_kinded (p : Join_env.parameters) (fuel : nat) :
  fresh_avoids p ->
  ∀ (equations : list (name * flambda_type)) (s s' : Join_env.t),
    state_valid s ->
    state_kinded s ->
    envs_invariant s ->
    envs_kinded s ->
    List.fold_left
      (fun env equation =>
         let* env := env in
         let* env :=
           Join_env.import_name_transitive p fuel env (fst equation)
         in
         Join_env.import_type_transitive p fuel env (snd equation))
      equations (Some s)
    = Some s' ->
    state_kinded s'.
Proof.
  intros Hfresh.
  induction equations as [ | [n ty] equations IH];
    intros s s' Hvalid Hkinded Hinvariant Henvs Hfold.
  - injection Hfold as <-. exact Hkinded.
  - cbn [List.fold_left fst snd] in Hfold.
    destruct (Join_env.import_name_transitive p fuel s n) as [s1 | ]
      eqn:Hname.
    + destruct (import_name_keeps _ _ _ _ _ Hfresh Hvalid Hname)
        as [Hvalid1 [Hgrows1 _]].
      pose proof (import_name_keeps_kinded _ _ _ _ _ Hfresh Hvalid Hkinded
                    Hinvariant Henvs Hname) as Hkinded1.
      pose proof (grows_envs_invariant _ _ Hgrows1 Hinvariant) as Hinvariant1.
      pose proof (grows_envs_kinded _ _ Hgrows1 Henvs) as Henvs1.
      destruct (Join_env.import_type_transitive p fuel s1 ty) as [s2 | ]
        eqn:Htype.
      * destruct (import_type_valid p fuel s1 s2 ty Hfresh Hvalid1 Htype)
          as [Hvalid2 [Hgrows2 _]].
        apply (IH s2 s' Hvalid2); try assumption.
        -- exact (import_vars_keeps_kinded _ (free_variables ty)
                    (import_var_keeps p fuel Hfresh)
                    (import_var_keeps_kinded p fuel Hfresh) s1 s2 Hvalid1
                    Hkinded1 Hinvariant1 Henvs1 Htype).
        -- exact (grows_envs_invariant _ _ Hgrows2 Hinvariant1).
        -- exact (grows_envs_kinded _ _ Hgrows2 Henvs1).
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
    + exfalso. clear -Hfold.
      induction equations as [ | e equations IH];
        [ discriminate | exact (IH Hfold) ].
Qed.

Lemma n_way_join_env_extension_kinds_in :
  ∀ (p : Join_env.parameters) (fuel : nat) (s : Join_env.t)
    (extensions : list (Index.t * env_extension))
    (result : Or_bottom.t (env_extension * Join_env.t)),
    fresh_avoids p ->
    state_valid s ->
    envs_invariant s ->
    envs_kinded s ->
    Join_env.n_way_join_env_extension p fuel s extensions = Some result ->
    state_kinded s ->
    join_args_in_scope Join_env.joined_env env_extension_kinded s extensions
    ->
    ∀ (joined : env_extension) (s' : Join_env.t),
      result = Or_bottom.Ok (joined, s') ->
      state_kinded s'
      ∧ env_extension_kinded_in_target Join_env.joined_env
          kind_in_target_env s' joined.
Proof.
  intros p fuel s extensions result Hfresh Hvalid Hinvariant Henvs Hjoin
    Hkinded Hscope joined s' Hresult.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ | ].
  { injection Hjoin as <-. discriminate. }
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as <-. injection Hresult as <- <-.
    destruct (Hscope index0 first (or_introl eq_refl))
      as [use0 [Huse0 Hfirst]].
    destruct first as [equations]. cbn [Join_env.equations_of] in Hfold.
    destruct (import_equations_keeps p fuel Hfresh _ _ _ Hvalid Hfold)
      as [Hvalid1 [Hgrows1 Hexists1]].
    pose proof (import_equations_keeps_kinded p fuel Hfresh _ _ _ Hvalid
                  Hkinded Hinvariant Henvs Hfold) as Hkinded1.
    split; [ exact Hkinded1 | ].
    intros simple kind Hmember.
    assert (Huse1 : Join_env.joined_env s1 index0 = Some use0).
    { destruct Hgrows1 as [Henvs1 _]. rewrite Henvs1. exact Huse0. }
    apply (simple_kind_in_target_of_use s1 index0 use0 simple kind Hvalid1
             Hkinded1 (grows_envs_invariant _ _ Hgrows1 Hinvariant)
             (grows_envs_kinded _ _ Hgrows1 Henvs) Huse1
             (Hfirst _ _ Hmember)).
    intros x Hx. apply Hexists1.
    exact (aliases_free_extension _ _ _ _ Hmember Hx).
  - injection Hjoin as <-. injection Hresult as <- <-.
    split; [ exact Hkinded | ].
    intros simple kind Hmember. destruct Hmember.
Qed.

End Kinds.

Theorem created_state_kinded : created_state_kinded_statement.
Proof.
  intros avoid source envs Hsource.
  split; [ exact Hsource | ]. split.
  - intros x kind index use Hx. discriminate.
  - intros x simples kind index simple use Hx. discriminate.
Qed.

Lemma simple_of_kind_in_scope (tenv : Typing_env.t) (simple : simple)
    (kind : flambda_kind) :
  simple_of_kind tenv simple kind -> simple_in_scope tenv simple.
Proof.
  intros Hkind x Hx. destruct simple as [y | sym | c]; try destruct Hx.
  - subst y. exact (simple_of_kind_var_known _ _ _ Hkind).
  - destruct H.
Qed.

Theorem n_way_join_simples_sound : n_way_join_simples_sound_statement.
Proof.
  intros p fuel Hfresh s s' kind simples result Hvalid Hwf Hkinded Hdistinct
    Hscope Hjoin.
  assert (Hscope' :
            join_args_in_scope Join_env.joined_env simple_in_scope s simples).
  { intros index simple Hin. destruct (Hscope _ _ Hin) as [use [Huse Hkind]].
    exists use. split; [ exact Huse | ].
    exact (simple_of_kind_in_scope _ _ _ Hkind). }
  destruct (n_way_join_simples_valid p fuel s s' kind simples result Hfresh
              Hvalid Hwf Hdistinct Hscope' Hjoin)
    as [Hvalid' [Hgrows [Hbottom Hupper]]].
  destruct (n_way_join_simples_kinds_in p fuel s s' kind simples result Hfresh
              Hvalid Hwf Hkinded Hdistinct Hscope Hjoin)
    as [Hsymbol Hkinds].
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ exact Hbottom | ]. split; [ exact Hsymbol | ].
  split; [ exact Hupper | exact Hkinds ].
Qed.

Theorem import_type_sound : import_type_sound_statement.
Proof.
  intros p fuel Hfresh s s' ty Hvalid Hwf Hkinded Himport.
  destruct (import_type_valid p fuel s s' ty Hfresh Hvalid Himport)
    as [Hvalid' [Hgrows Htarget]].
  split; [ exact Hvalid' | ]. split; [ exact Hgrows | ].
  split; [ exact Htarget | ].
  intros Hstate Hty.
  exact (import_type_kinds_in (proj1 aliases_free_variables) p fuel s s' ty
           Hfresh Hvalid Hwf Hkinded Himport Hstate Hty).
Qed.

Theorem n_way_join_env_extension_sound
  : n_way_join_env_extension_sound_statement.
Proof.
  intros p fuel Hfresh Heqb s extensions result Hvalid Hwf Hkinded Hdistinct
    Hscope Hjoin.
  destruct (n_way_join_env_extension_valid p fuel s extensions result Hfresh
              Heqb Hvalid Hscope Hjoin) as [Hbottom Hok].
  split; [ exact Hbottom | ]. split; [ exact Hok | ].
  intros Hstate Hextensions joined s' Hresult.
  exact (n_way_join_env_extension_kinds_in (proj2 aliases_free_variables) p
           fuel s extensions result Hfresh Hvalid Hwf Hkinded Hjoin Hstate
           Hextensions joined s' Hresult).
Qed.

(* ---------------------------------------------------------------------- *)
(* The names of the equations inside what is answered                      *)
(* ---------------------------------------------------------------------- *)

Theorem import_type_equations_kinded : import_type_equations_kinded_statement.
Proof.
  intros p fuel Hfresh s s' ty Hvalid Hinvariant Henvs Hkinded
    [index [use [Huse Hty]]] Himport.
  destruct (import_type_valid p fuel s s' ty Hfresh Hvalid Himport)
    as [Hvalid' [Hgrows' Htarget]].
  pose proof (import_vars_keeps_kinded _ (free_variables ty)
                (import_var_keeps p fuel Hfresh)
                (import_var_keeps_kinded p fuel Hfresh) s s' Hvalid Hkinded
                Hinvariant Henvs Himport) as Hkinded'.
  intros n kind Hmember.
  assert (Huse' : Join_env.joined_env s' index = Some use).
  { destruct Hgrows' as [Henvs' _]. rewrite Henvs'. exact Huse. }
  apply (simple_kind_in_target_of_use s' index use (simple_of_name n) kind
           Hvalid' Hkinded' (grows_envs_invariant _ _ Hgrows' Hinvariant)
           (grows_envs_kinded _ _ Hgrows' Henvs) Huse' (Hty _ _ Hmember)).
  intros x Hx. apply Htarget.
  exact (proj1 equation_names_free_variables ty n kind x Hmember Hx).
Qed.

Theorem n_way_join_env_extension_equations_kinded
  : n_way_join_env_extension_equations_kinded_statement.
Proof.
  intros p fuel Hfresh s s' extensions joined Hvalid Hinvariant Henvs Hkinded
    Hscope Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as <- <-.
    destruct (Hscope index0 first (or_introl eq_refl))
      as [use0 [Huse0 Hfirst]].
    pose proof Hfold as Hfold'. destruct first as [equations].
    cbn [Join_env.equations_of] in Hfold'.
    destruct (import_equations_keeps p fuel Hfresh _ _ _ Hvalid Hfold')
      as [Hvalid1 [Hgrows1 Hexists1]].
    pose proof (import_equations_keeps_kinded p fuel Hfresh _ _ _ Hvalid
                  Hkinded Hinvariant Henvs Hfold') as Hkinded1.
    intros n kind Hmember.
    assert (Huse1 : Join_env.joined_env s1 index0 = Some use0).
    { destruct Hgrows1 as [Henvs1 _]. rewrite Henvs1. exact Huse0. }
    apply (simple_kind_in_target_of_use s1 index0 use0 (simple_of_name n)
             kind Hvalid1 Hkinded1 (grows_envs_invariant _ _ Hgrows1 Hinvariant)
             (grows_envs_kinded _ _ Hgrows1 Henvs) Huse1
             (Hfirst _ _ Hmember)).
    intros x Hx. apply Hexists1.
    exact (proj2 equation_names_free_variables _ n kind x Hmember Hx).
  - injection Hjoin as <- <-. intros n kind Hmember. destruct Hmember.
Qed.

(* ======================================================================== *)
(* The definitions are kept at every state                                  *)
(* ======================================================================== *)

Lemma same_readings_keeps (s s' : Join_env.t) :
  same_readings s s' -> keeps_definitions s s'.
Proof.
  destruct s, s'. unfold same_readings. cbn. intros [-> [_ [-> _]]].
  split; [ intros index; reflexivity | intros x definition H; exact H ].
Qed.

Lemma existential_keeps_definitions (p : Join_env.parameters)
    (s s' : Join_env.t) (canonicals : Simples_in_joined_envs.t)
    (kind : flambda_kind) (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  keeps_definitions s s'.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0].
  injection Hmake as _ <-.
  split; [ intros index; reflexivity | ].
  intros y definition Hy. cbn [Join_env.definitions_of_existentials].
  destruct (is_bound x0 (Join_env.definitions_of_existentials s)) eqn:Hbound;
    [ exact Hy | ].
  cbn [lookup]. destruct (Pos.eqb y x0) eqn:Heq; [ | exact Hy ].
  apply Pos.eqb_eq in Heq. subst y. unfold is_bound in Hbound.
  rewrite Hy in Hbound. discriminate.
Qed.

Definition import_keeps_definitions
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    import_var s x = Some s' -> keeps_definitions s s'.

Lemma import_vars_keeps_definitions
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps_definitions import_var ->
  ∀ s s' : Join_env.t,
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    keeps_definitions s s'.
Proof.
  intros Himport.
  induction vars as [ | var vars IH]; intros s s' Hfold.
  - injection Hfold as <-. apply keeps_definitions_refl.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + exact (keeps_definitions_trans _ _ _ (Himport _ _ _ Hvar)
               (IH _ _ Hfold)).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equation_keeps_definitions (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  keeps_definitions s s'.
Proof.
  intros Hadd.
  exact (same_readings_keeps _ _ (add_equation_same_readings _ _ _ _ Hadd)).
Qed.

Lemma add_alias_keeps_definitions (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  keeps_definitions s
    (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  apply (keeps_definitions_trans _ _ _
           (existential_keeps_definitions _ _ _ _ _ _ Hmake)).
  apply same_readings_keeps. repeat split.
Qed.

Lemma add_equals_keeps_definitions (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  import_keeps_definitions import_var ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  keeps_definitions s s'.
Proof.
  intros Himport Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_keeps_definitions _ _ _ _ Hadd).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    exact (keeps_definitions_trans _ _ _ (Himport _ _ _ Hvar)
             (add_equation_keeps_definitions _ _ _ _ Hadd)).
  - injection Hadd as <-. apply add_alias_keeps_definitions.
Qed.

Lemma concrete_keeps_definitions (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps_definitions import_var ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  keeps_definitions s s'.
Proof.
  intros Himport Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    exact (keeps_definitions_trans _ _ _
             (add_equation_keeps_definitions _ _ _ _ Hadd)
             (import_vars_keeps_definitions _ _ Himport _ _ Hconcrete)).
  - injection Hconcrete as <-. apply same_readings_keeps.
    apply add_equations_same_readings.
Qed.

Lemma import_body_keeps_definitions (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_keeps_definitions import_var ->
  import_keeps_definitions (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Himport s s' x Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x);
    [ injection Hbody as <-; apply keeps_definitions_refl | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; apply keeps_definitions_refl | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ]; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Hfirst : keeps_definitions s (with_import s x kind)).
  { split; [ intros index; reflexivity | intros y definition H; exact H ]. }
  apply (keeps_definitions_trans _ _ _ Hfirst).
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ];
    [ | discriminate ].
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ]; try discriminate.
    exact (add_equals_keeps_definitions _ _ _ _ _ _ _ Himport Hbody).
  - exact (concrete_keeps_definitions _ _ _ _ _ _ Himport Hbody).
Qed.

Lemma import_var_keeps_definitions (p : Join_env.parameters) (fuel : nat) :
  import_keeps_definitions (Join_env.import_var_transitive p fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' x H. discriminate.
  - exact (import_body_keeps_definitions p _ IH).
Qed.

Theorem import_type_keeps_definitions
  : import_type_keeps_definitions_statement.
Proof.
  intros p fuel s s' ty Himport.
  exact (import_vars_keeps_definitions _ (free_variables ty)
           (import_var_keeps_definitions p fuel) s s' Himport).
Qed.

Theorem n_way_join_simples_keeps_definitions
  : n_way_join_simples_keeps_definitions_statement.
Proof.
  intros p fuel s s' kind simples result Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. apply keeps_definitions_refl. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. apply keeps_definitions_refl.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_keeps_definitions p fuel _ _ _ Himport).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_keeps_definitions _ _ _ _ _ _ Hmake).
Qed.

Theorem n_way_join_env_extension_keeps_definitions
  : n_way_join_env_extension_keeps_definitions_statement.
Proof.
  intros p fuel s s' extensions joined Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply keeps_definitions_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : keeps_definitions s s2).
        { destruct n as [x | sym].
          - exact (import_var_keeps_definitions p fuel _ _ _ Hname).
          - injection Hname as <-. apply keeps_definitions_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (keeps_definitions_trans _ _ _ Hfirst
                    (keeps_definitions_trans _ _ _
                       (import_type_keeps_definitions p fuel _ _ _ Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply keeps_definitions_refl.
Qed.

(* ---------------------------------------------------------------------- *)
(* Two facts for the others                                                *)
(* ---------------------------------------------------------------------- *)

Theorem variables_of_env_knows : variables_of_env_knows_statement.
Proof.
  intros tenv x. unfold Join_env.variables_of_env, knows.
  rewrite mem_var_lookup. split.
  - intros Hin. apply in_flat_map in Hin.
    destruct Hin as [[n [ty btm]] [Hin Hx]]. cbn [fst] in Hx.
    destruct n as [y | sym]; [ | destruct Hx ].
    destruct Hx as [<- | []].
    destruct (Keyed.lookup name_eqb (Name_var y)
                (Typing_env.names_to_types tenv)) as [[ty' btm'] | ]
      eqn:Hlookup.
    + exists ty', btm'. reflexivity.
    + exfalso.
      apply (keyed_lookup_none_not_in name_eqb name_eqb_decides_equality _ _
               Hlookup).
      apply in_map_iff. exists (Name_var y, (ty, btm)).
      split; [ reflexivity | exact Hin ].
  - intros [ty [btm Hlookup]]. apply in_flat_map.
    exists (Name_var x, (ty, btm)).
    split; [ | left; reflexivity ].
    exact (keyed_lookup_in name_eqb name_eqb_decides_equality _ _ _ Hlookup).
Qed.

Theorem n_way_join_env_extension_answers_given
  : n_way_join_env_extension_answers_given_statement.
Proof.
  intros p fuel s s' extensions extension Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ]; [ | discriminate ].
    injection Hjoin as <- _. right. exists index0. left. reflexivity.
  - injection Hjoin as <- _. left. reflexivity.
Qed.
