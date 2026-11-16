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


(* The proofs of the statements of FlambdaJoinAddsSpec.v. This file
   needs no review.

   The last phase is its fold over the variables that the join has
   made, then the rest ([last_phase_unfold]). The fold with any
   functions goes as the fold that records ([define_fold_transfer]):
   the environment of the first is the environment given with the
   variables recorded by the second defined in it. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeProject FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinEnvProofs FlambdaJoinStateSpec
  FlambdaJoinStateProofs FlambdaJoinDefineSpec FlambdaJoinDefineProofs
  FlambdaJoinFitsSpec FlambdaJoinFitsProofs FlambdaJoinAddsSpec.
Import ListNotations.

Definition unique_of (s : Join_env.t) (relations : Join_env.inverse_relations)
    (var : variable) : bool :=
  negb (Join_env.in_inverse_relations relations var)
  && Nat.leb
       (Join_env.count_occurrences var
          (flat_map
             (fun binding : variable * flambda_type =>
                free_variables (snd binding))
             (compact (Join_env.types_in_target_env s))))
       1.

Definition last_phase_tail {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (mh : meet_expanded_head)
    (acc : option (define_acc meet_env))
    : option
        (meet_env * finite_map (Simples_in_joined_envs.t * flambda_kind)) :=
  let* '(target_env, to_expand, equations, relations, definitions) := acc in
  let to_project := domain to_expand in
  let expand := Join_env.expand_variable fuel to_expand in
  let* equations :=
    traverse
      (fun binding =>
         let* ty := project_variables_out to_project expand (snd binding) in
         Some (Name_var (fst binding), ty))
      (compact equations)
  in
  let* equations_for_inverse_relations :=
    traverse
      (fun binding =>
         let* ty :=
           project_variables_out to_project expand
             (Type_naked_immediate
                (Ok
                   (No_alias
                      (Head_of_kind_naked_immediate Or_unknown.Unknown
                         (snd binding)))))
         in
         Some (Name_var (fst binding), ty))
      (compact relations)
  in
  let* target_env :=
    Join_env.add_env_extension ops target_env (Env_extension equations) mh
  in
  let* target_env :=
    Join_env.add_env_extension ops target_env
      (Env_extension equations_for_inverse_relations) mh
  in
  Some (target_env, definitions).

Lemma last_phase_unfold {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (mh : meet_expanded_head) (s : Join_env.t) (menv : meet_env)
    (relations : Join_env.inverse_relations) :
  Join_env.define_or_eliminate_variables_and_add_equations ops fuel mh s menv
    relations
  = last_phase_tail ops fuel mh
      (List.fold_left
         (define_step ops s
            (Join_env.current_typing_env ops
               (variables_defined ops menv
                  (Bindings_in_target_env.imported_variables_in_order
                     (Join_env.bindings s))))
            (unique_of s relations))
         (Bindings_in_target_env.existential_variables_in_order
            (Join_env.bindings s))
         (Some
            (variables_defined ops menv
               (Bindings_in_target_env.imported_variables_in_order
                  (Join_env.bindings s)),
             [], Join_env.types_in_target_env s, relations,
             Join_env.definitions_of_existentials s))).
Proof. reflexivity. Qed.

Lemma recording_defined (tenv : Typing_env.t) :
  ∀ (vars : list (variable * flambda_kind)) (r : recorded),
    variables_defined (recording tenv) r vars
    = {| recorded_variables := recorded_variables r ++ vars;
         recorded_extensions := recorded_extensions r |}.
Proof.
  induction vars as [ | [x k] vars IH]; intros r.
  - cbn. rewrite app_nil_r. destruct r. reflexivity.
  - change (variables_defined (recording tenv) r ((x, k) :: vars))
      with (variables_defined (recording tenv)
              {| recorded_variables := recorded_variables r ++ [(x, k)];
                 recorded_extensions := recorded_extensions r |}
              vars).
    rewrite IH. cbn [recorded_variables recorded_extensions].
    rewrite <- app_assoc. reflexivity.
Qed.

(* The state of the fold for any functions, from that of the fold that
   records. *)
Definition transfer {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (menv : meet_env) (acc : define_acc recorded) : define_acc meet_env :=
  let '(r, to_expand, equations, relations, definitions) := acc in
  (variables_defined ops menv (recorded_variables r), to_expand, equations,
   relations, definitions).

Lemma define_step_transfer {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (menv : meet_env) (tenv : Typing_env.t) (s : Join_env.t)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool)
    (acc : option (define_acc recorded)) (binding : variable * flambda_kind) :
  define_step ops s tenv_of_aliases unique (option_map (transfer ops menv) acc)
    binding
  = option_map (transfer ops menv)
      (define_step (recording tenv) s tenv_of_aliases unique acc binding).
Proof.
  destruct acc as [[[[[r to_expand] equations] relations] definitions] | ];
    [ | reflexivity ].
  cbn [option_map transfer define_step].
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ];
    [ | | reflexivity ].
  - destruct (Join_env.move_equation equations (fst binding) canon_var);
      reflexivity.
  - destruct (unique (fst binding)); [ reflexivity | ].
    cbn [option_map transfer Join_env.add_variable_definition recording
           recorded_variables].
    unfold variables_defined. rewrite fold_left_app. reflexivity.
Qed.

Lemma define_fold_transfer {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (menv : meet_env) (tenv : Typing_env.t) (s : Join_env.t)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool) :
  ∀ (l : list (variable * flambda_kind))
    (acc : option (define_acc recorded)),
    List.fold_left (define_step ops s tenv_of_aliases unique) l
      (option_map (transfer ops menv) acc)
    = option_map (transfer ops menv)
        (List.fold_left (define_step (recording tenv) s tenv_of_aliases unique)
           l acc).
Proof.
  induction l as [ | binding l IH]; intros acc; [ reflexivity | ].
  cbn [List.fold_left]. rewrite (define_step_transfer ops menv tenv).
  apply IH.
Qed.

(* What the fold that records has recorded. *)
Lemma recording_fold_kept (tenv : Typing_env.t) (s : Join_env.t)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool) :
  ∀ (l : list (variable * flambda_kind)) (acc acc' : define_acc recorded),
    List.fold_left (define_step (recording tenv) s tenv_of_aliases unique) l
      (Some acc)
    = Some acc' ->
    ∃ kept : list (variable * flambda_kind),
      recorded_variables (acc_env acc')
      = recorded_variables (acc_env acc) ++ kept
      ∧ recorded_extensions (acc_env acc') = recorded_extensions (acc_env acc)
      ∧ (∀ binding : variable * flambda_kind, binding ∈ kept -> binding ∈ l)
      ∧ (NoDup (map fst l) -> NoDup (map fst kept)).
Proof.
  induction l as [ | binding l IH]; intros acc acc' Hfold.
  - injection Hfold as <-. exists [].
    split; [ rewrite app_nil_r; reflexivity | ]. split; [ reflexivity | ].
    split; [ intros b [] | intros _; apply NoDup_nil ].
  - cbn [List.fold_left] in Hfold.
    destruct (define_step (recording tenv) s tenv_of_aliases unique (Some acc)
                binding) as [acc1 | ] eqn:Hstep.
    2:{ rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold.
        discriminate. }
    destruct (IH acc1 acc' Hfold) as [kept [Hvars [Hexts [Hin Hdistinct]]]].
    destruct acc as [[[[r to_expand] equations] relations] definitions].
    cbn [define_step] in Hstep.
    assert (Hsame :
              acc_env acc1 = r ->
              ∃ kept0 : list (variable * flambda_kind),
                recorded_variables (acc_env acc')
                = recorded_variables r ++ kept0
                ∧ recorded_extensions (acc_env acc') = recorded_extensions r
                ∧ (∀ b : variable * flambda_kind,
                     b ∈ kept0 -> b ∈ binding :: l)
                ∧ (NoDup (map fst (binding :: l)) -> NoDup (map fst kept0))).
    { intros Hr. exists kept. rewrite Hvars, Hexts, Hr.
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ intros b Hb; right; exact (Hin b Hb) | ].
      intros Hnodup. cbn [map] in Hnodup. apply NoDup_cons_iff in Hnodup.
      exact (Hdistinct (proj2 Hnodup)). }
    destruct (Join_env.choose_alias _ _) as [[canon_var | ] | ];
      [ | | discriminate ].
    + destruct (Join_env.move_equation equations (fst binding) canon_var);
        [ | discriminate ].
      injection Hstep as <-. exact (Hsame eq_refl).
    + destruct (unique (fst binding)).
      * injection Hstep as <-. exact (Hsame eq_refl).
      * injection Hstep as <-. exists (binding :: kept).
        unfold acc_env in Hvars, Hexts |- *. cbn [fst] in Hvars, Hexts |- *.
        cbn [Join_env.add_variable_definition recording recorded_variables
               recorded_extensions] in Hvars, Hexts.
        split.
        { rewrite Hvars, <- app_assoc. destruct binding. reflexivity. }
        split; [ exact Hexts | ].
        split.
        { intros b [<- | Hb];
            [ left; reflexivity | right; exact (Hin b Hb) ]. }
        intros Hnodup. cbn [map] in Hnodup |- *.
        apply NoDup_cons_iff in Hnodup. destruct Hnodup as [Hnotin Hnodup].
        apply NoDup_cons; [ | exact (Hdistinct Hnodup) ].
        intros Hfst. apply Hnotin. apply in_map_iff in Hfst.
        destruct Hfst as [b [Heq Hb]]. apply in_map_iff. exists b.
        split; [ exact Heq | exact (Hin b Hb) ].
Qed.

Lemma in_order_distinct {A : Type} (m : finite_map A) :
  NoDup (map fst (rev (compact m))).
Proof.
  rewrite map_rev. apply NoDup_rev. apply distinct_NoDup.
  apply compact_distinct.
Qed.

(* The run that records, in parts. *)
Lemma last_phase_adds_parts (tenv : Typing_env.t) (fuel : nat)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (r : recorded)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  Join_env.define_or_eliminate_variables_and_add_equations (recording tenv)
    fuel tt s nothing_recorded relations
  = Some (r, definitions) ->
  ∃ (r1 : recorded) (to_expand equations1 : finite_map flambda_type)
    (relations1 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type)),
    List.fold_left
      (define_step (recording tenv) s tenv (unique_of s relations))
      (Bindings_in_target_env.existential_variables_in_order
         (Join_env.bindings s))
      (Some
         ({| recorded_variables :=
               Bindings_in_target_env.imported_variables_in_order
                 (Join_env.bindings s);
             recorded_extensions := [] |},
          [], Join_env.types_in_target_env s, relations,
          Join_env.definitions_of_existentials s))
    = Some (r1, to_expand, equations1, relations1, definitions)
    ∧ recorded_variables r1
      = Bindings_in_target_env.imported_variables_in_order
          (Join_env.bindings s)
        ++ kept
    ∧ recorded_extensions r1 = []
    ∧ NoDup (map fst kept)
    ∧ (∀ binding : variable * flambda_kind,
         binding ∈ kept ->
         binding
         ∈ Bindings_in_target_env.existential_variables_in_order
             (Join_env.bindings s))
    ∧ traverse
        (fun binding =>
           let* ty :=
             project_variables_out (domain to_expand)
               (Join_env.expand_variable fuel to_expand) (snd binding)
           in
           Some (Name_var (fst binding), ty))
        (compact equations1)
      = Some equations
    ∧ traverse
        (fun binding =>
           let* ty :=
             project_variables_out (domain to_expand)
               (Join_env.expand_variable fuel to_expand)
               (Type_naked_immediate
                  (Ok
                     (No_alias
                        (Head_of_kind_naked_immediate Or_unknown.Unknown
                           (snd binding)))))
           in
           Some (Name_var (fst binding), ty))
        (compact relations1)
      = Some relation_equations
    ∧ r
      = {| recorded_variables :=
             Bindings_in_target_env.imported_variables_in_order
               (Join_env.bindings s)
             ++ kept;
           recorded_extensions :=
             [Env_extension equations; Env_extension relation_equations] |}.
Proof.
  intros Hrun. rewrite last_phase_unfold in Hrun.
  rewrite recording_defined in Hrun.
  cbn [Join_env.current_typing_env recording nothing_recorded
         recorded_variables recorded_extensions app] in Hrun.
  match type of Hrun with
  | last_phase_tail _ _ _ ?fold = _ =>
    destruct fold
      as [[[[[r1 to_expand] equations1] relations1] definitions1] | ]
      eqn:Hfold; [ | discriminate ]
  end.
  destruct (recording_fold_kept tenv s tenv (unique_of s relations) _ _ _
              Hfold) as [kept [Hvars [Hexts [Hin Hdistinct]]]].
  unfold acc_env in Hvars, Hexts.
  cbn [fst recorded_variables recorded_extensions] in Hvars, Hexts.
  unfold last_phase_tail in Hrun.
  match type of Hrun with
  | context [traverse ?f (compact equations1)] =>
    destruct (traverse f (compact equations1)) as [equations | ]
      eqn:Hequations; [ | discriminate ]
  end.
  match type of Hrun with
  | context [traverse ?f (compact relations1)] =>
    destruct (traverse f (compact relations1)) as [relation_equations | ]
      eqn:Hrelation_equations; [ | discriminate ]
  end.
  cbn [Join_env.add_env_extension recording recorded_variables
         recorded_extensions] in Hrun.
  injection Hrun as <- <-.
  exists r1, to_expand, equations1, relations1, kept, equations,
    relation_equations.
  split; [ reflexivity | ]. split; [ exact Hvars | ].
  split; [ exact Hexts | ].
  split; [ exact (Hdistinct (in_order_distinct _)) | ].
  split; [ exact Hin | ]. split; [ exact Hequations | ].
  split; [ exact Hrelation_equations | ].
  rewrite Hvars, Hexts. reflexivity.
Qed.

Lemma recording_witnesses (tenv : Typing_env.t)
    (imported kept kept' : list (variable * flambda_kind))
    (equations equations' relation_equations relation_equations'
     : list (name * flambda_type))
    (menv' : recorded) :
  Join_env.add_env_extension (recording tenv)
    (variables_defined (recording tenv) nothing_recorded (imported ++ kept'))
    (Env_extension equations') tt
  = Some menv' ->
  Join_env.add_env_extension (recording tenv) menv'
    (Env_extension relation_equations') tt
  = Some
      {| recorded_variables := imported ++ kept;
         recorded_extensions :=
           [Env_extension equations; Env_extension relation_equations] |} ->
  kept' = kept
  ∧ equations' = equations
  ∧ relation_equations' = relation_equations.
Proof.
  intros H1 H2. rewrite recording_defined in H1.
  cbn [Join_env.add_env_extension recording nothing_recorded
         recorded_variables recorded_extensions app] in H1.
  injection H1 as <-.
  cbn [Join_env.add_env_extension recording recorded_variables
         recorded_extensions app] in H2.
  injection H2 as Hvars Hequations Hrelations.
  apply app_inv_head in Hvars. subst.
  split; [ reflexivity | split; reflexivity ].
Qed.

Theorem last_phase_adds_functional : last_phase_adds_functional_statement.
Proof.
  intros tenv fuel s relations kept kept' equations equations'
    relation_equations relation_equations' definitions definitions' H H'.
  unfold last_phase_adds in H, H'. rewrite H in H'.
  injection H' as Hvars Hequations Hrelations Hdefinitions.
  apply app_inv_head in Hvars. subst.
  split; [ reflexivity | ]. split; [ reflexivity | split; reflexivity ].
Qed.

Theorem last_phase_adds_kept : last_phase_adds_kept_statement.
Proof.
  intros tenv fuel s relations kept equations relation_equations definitions
    Hadds.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [_ [_ [_ [Hdistinct [Hin [_ [_ Heq]]]]]]]]]]]]]].
  injection Heq as Hvars _ _. apply app_inv_head in Hvars. subst kept0.
  split; [ exact Hdistinct | exact Hin ].
Qed.

Lemma define_fold_transfer_start {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (menv : meet_env) (tenv : Typing_env.t) (s : Join_env.t)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool)
    (l vars : list (variable * flambda_kind))
    (to_expand equations : finite_map flambda_type)
    (relations : Join_env.inverse_relations)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)) :
  List.fold_left (define_step ops s tenv_of_aliases unique) l
    (Some
       (variables_defined ops menv vars, to_expand, equations, relations,
        definitions))
  = option_map (transfer ops menv)
      (List.fold_left (define_step (recording tenv) s tenv_of_aliases unique)
         l
         (Some
            ({| recorded_variables := vars; recorded_extensions := [] |},
             to_expand, equations, relations, definitions))).
Proof.
  exact (define_fold_transfer ops menv tenv s tenv_of_aliases unique l
           (Some
              ({| recorded_variables := vars; recorded_extensions := [] |},
               to_expand, equations, relations, definitions))).
Qed.

Theorem last_phase_replayed : last_phase_replayed_statement.
Proof.
  intros meet_env meet_expanded_head ops fuel mh s menv target relations
    definitions Hdefine.
  rewrite last_phase_unfold in Hdefine.
  rewrite (define_fold_transfer_start ops menv
             (Join_env.current_typing_env ops
                (variables_defined ops menv
                   (Bindings_in_target_env.imported_variables_in_order
                      (Join_env.bindings s)))))
    in Hdefine.
  match type of Hdefine with
  | last_phase_tail _ _ _ (option_map _ ?fold) = _ =>
    destruct fold
      as [[[[[r1 to_expand] equations1] relations1] definitions1] | ]
      eqn:Hfold; [ | discriminate ]
  end.
  cbn [option_map transfer] in Hdefine. unfold last_phase_tail in Hdefine.
  match type of Hdefine with
  | context [traverse ?f (compact equations1)] =>
    destruct (traverse f (compact equations1)) as [equations | ]
      eqn:Hequations; [ | discriminate ]
  end.
  match type of Hdefine with
  | context [traverse ?f (compact relations1)] =>
    destruct (traverse f (compact relations1)) as [relation_equations | ]
      eqn:Hrelation_equations; [ | discriminate ]
  end.
  destruct (Join_env.add_env_extension ops
              (variables_defined ops menv (recorded_variables r1))
              (Env_extension equations) mh)
    as [menv2 | ] eqn:Hadd1; [ | discriminate ].
  destruct (Join_env.add_env_extension ops menv2
              (Env_extension relation_equations) mh) as [menv3 | ] eqn:Hadd2;
    [ | discriminate ].
  injection Hdefine as <- <-.
  destruct (recording_fold_kept _ s _ (unique_of s relations) _ _ _ Hfold)
    as [kept [Hvars [Hexts _]]].
  unfold acc_env in Hvars, Hexts.
  cbn [fst recorded_variables recorded_extensions] in Hvars, Hexts.
  exists kept, equations, relation_equations, menv2.
  split; [ | split; [ rewrite <- Hvars; exact Hadd1 | exact Hadd2 ] ].
  unfold last_phase_adds. rewrite last_phase_unfold, recording_defined.
  cbn [Join_env.current_typing_env recording nothing_recorded
         recorded_variables recorded_extensions app].
  etransitivity;
    [ exact (f_equal (last_phase_tail (recording _) fuel tt) Hfold) | ].
  unfold last_phase_tail. rewrite Hequations, Hrelation_equations.
  cbn [Join_env.add_env_extension recording recorded_variables
         recorded_extensions app].
  rewrite Hvars, Hexts. reflexivity.
Qed.

Theorem last_phase_adds_true : last_phase_adds_true_statement.
Proof.
  intros tenv fuel source s relations kept equations relation_equations
    definitions Htrue Hkinded Henvs Hrelations Hnonempty Hadds.
  destruct (define_variables_true recorded unit (recording tenv) fuel tt
              source s nothing_recorded _ relations definitions Htrue Hkinded
              Henvs Hrelations Hnonempty Hadds)
    as [kept' [equations' [relation_equations' [menv' [_ [Hadd1 [Hadd2
          Hruns]]]]]]].
  destruct (recording_witnesses _ _ _ _ _ _ _ _ _ Hadd1 Hadd2)
    as [-> [-> ->]].
  exact Hruns.
Qed.

Theorem last_phase_adds_in_scope : last_phase_adds_in_scope_statement.
Proof.
  intros tenv fuel source s relations kept equations relation_equations
    definitions Htrue Hrelations Hnonempty Htarget Hnot_made Hadds.
  destruct (define_variables_in_scope recorded unit (recording tenv) fuel tt
              source s nothing_recorded _ relations definitions Htrue
              Hrelations Hnonempty Htarget Hnot_made Hadds)
    as [kept' [equations' [relation_equations' [menv' [_ [Hadd1 [Hadd2
          [Hscope1 Hscope2]]]]]]]].
  destruct (recording_witnesses _ _ _ _ _ _ _ _ _ Hadd1 Hadd2)
    as [-> [-> ->]].
  split; [ exact Hscope1 | exact Hscope2 ].
Qed.

Theorem last_phase_adds_on_variables
  : last_phase_adds_on_variables_statement.
Proof.
  intros tenv fuel s relations kept equations relation_equations definitions
    Hadds n ty Hin.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [_ [_ [_ [_ [_ [Hequations [Hrelation_equations Heq]]]]]]]]]]]]]].
  injection Heq as _ -> ->. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - destruct (traverse_in _ _ _ Hequations (n, ty) Hin) as [[x ty0] [_ Hf]].
    cbn [fst snd] in Hf.
    destruct (project_variables_out _ _ ty0); [ | discriminate ].
    injection Hf as <- _. exists x. reflexivity.
  - destruct (traverse_in _ _ _ Hrelation_equations (n, ty) Hin)
      as [[x names] [_ Hf]].
    cbn [fst snd] in Hf.
    match type of Hf with
    | context [project_variables_out ?to_project ?expand ?ty] =>
      destruct (project_variables_out to_project expand ty);
        [ | discriminate ]
    end.
    injection Hf as <- _. exists x. reflexivity.
Qed.

Theorem cut_and_n_way_join0_replayed : cut_and_n_way_join0_replayed_statement.
Proof.
  intros meet_env meet_expanded_head ops p fuel choices jt mh avoid menv
    target source envs definitions names Hempty Hjoin.
  unfold Join_env.cut_and_n_way_join0 in Hjoin. rewrite Hempty in Hjoin.
  cbv zeta in Hjoin.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  destruct (Join_env.join_aliases_into_bindings p fuel s0 envs) as [s1 | ]
    eqn:Haliases; [ | discriminate ].
  destruct (Join_env.n_way_join_delayed_equations fuel choices jt s1 []
              (Join_env.new_equations_in_joined_envs s1 s0))
    as [[relations2 s2] | ] eqn:Hdelayed; [ | discriminate ].
  destruct (Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices
              jt s2 relations2
              (Join_env.new_definitions_of_existentials s2 s0))
    as [[relations3 s3] | ] eqn:Hloop; [ | discriminate ].
  destruct (Join_env.define_or_eliminate_variables_and_add_equations ops fuel
              mh s3 _ relations3) as [[target' definitions'] | ]
    eqn:Hdefine; [ | discriminate ].
  injection Hjoin as <- <- <-.
  destruct (last_phase_replayed meet_env meet_expanded_head ops fuel mh s3
              _ target' relations3 definitions' Hdefine)
    as [kept [equations [relation_equations [menv' [Hadds [Hadd1 Hadd2]]]]]].
  exists s1, s2, s3, relations2, relations3, kept, equations,
    relation_equations, menv'.
  split; [ reflexivity | ]. split; [ exact Hdelayed | ].
  split; [ exact Hloop | ]. split; [ reflexivity | ].
  split; [ exact Hadds | ]. split; [ exact Hadd1 | exact Hadd2 ].
Qed.

(* The definitions, through the fold of the last phase: each is of a
   variable of the target so far that is not being removed; and the
   variables that are being removed are variables that the join has
   made. *)
Definition definitions_invariant {meet_env : Type} (s : Join_env.t)
    (acc : define_acc meet_env) : Prop :=
  let '(_, to_expand, _, _, definitions) := acc in
  (∀ (e : variable) (ty : flambda_type),
     lookup e to_expand = Some ty -> is_existential s e)
  ∧ ∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
      lookup x definitions = Some definition ->
      exists_in_target s x = true ∧ is_bound x to_expand = false.

Lemma define_step_definitions {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (source : Typing_env.t) (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
    (unique : variable -> bool) (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  state_true source s ->
  aliases_not_made s ->
  binding
  ∈ Bindings_in_target_env.existential_variables_in_order (Join_env.bindings s)
  ->
  definitions_invariant s acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  definitions_invariant s acc'.
Proof.
  intros Htrue Hnot_made Hbinding Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hexpand Hdefs].
  assert (Hvar : is_existential s var).
  { apply is_bound_lookup. exists var_kind.
    exact (in_order_lookup _ _ _ Hbinding). }
  assert (Hnot_expanded :
            ∀ x : variable,
              ~ is_existential s x -> is_bound x to_expand = false).
  { intros x Hx. destruct (is_bound x to_expand) eqn:Hb; [ | reflexivity ].
    apply is_bound_lookup in Hb. destruct Hb as [ty Hty].
    exfalso. exact (Hx (Hexpand x ty Hty)). }
  assert (Hne : ∀ x : variable, ~ is_existential s x -> Pos.eqb x var = false).
  { intros x Hx. destruct (Pos.eqb x var) eqn:Heq; [ | reflexivity ].
    apply Pos.eqb_eq in Heq. subst x. exfalso. exact (Hx Hvar). }
  assert (Hexpand1 :
            ∀ (ty0 : flambda_type) (e : variable) (ty : flambda_type),
              lookup e ((var, ty0) :: to_expand) = Some ty ->
              is_existential s e).
  { intros ty0 e ty He. rewrite lookup_cons in He.
    destruct (Pos.eqb e var) eqn:Heq; [ | exact (Hexpand e ty He) ].
    apply Pos.eqb_eq in Heq. subst e. exact Hvar. }
  assert (Hremoved :
            ∀ (ty0 : flambda_type) (x : variable)
              (definition : Simples_in_joined_envs.t * flambda_kind),
              lookup x (unbind var definitions) = Some definition ->
              exists_in_target s x = true
              ∧ is_bound x ((var, ty0) :: to_expand) = false).
  { intros ty0 x d Hx. rewrite lookup_unbind in Hx.
    destruct (Pos.eqb x var) eqn:Hxv; [ discriminate | ].
    destruct (Hdefs x d Hx) as [Hb Hn]. split; [ exact Hb | ].
    rewrite is_bound_cons, Hxv, Hn. reflexivity. }
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ]
    eqn:Hchoose; [ | | discriminate ].
  - (* an alias in the target stands for the variable *)
    destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ] eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    pose proof (choose_alias_in _ _ _ Hchoose) as Hcanon_in.
    assert (Hcanon_base : exists_in_target s canon_var = true).
    { destruct Htrue as [_ [[_ [_ [Haliases _]]] _]].
      exact (proj1 (Haliases var canon_var Hcanon_in)). }
    pose proof (Hnot_made var canon_var Hcanon_in) as Hcanon_not.
    split; [ exact (Hexpand1 _) | ].
    intros x d Hx. unfold Join_env.move_definition in Hx.
    destruct (lookup var definitions) as [defn | ] eqn:Hvar_def.
    + rewrite lookup_cons in Hx. destruct (Pos.eqb x canon_var) eqn:Heq.
      * apply Pos.eqb_eq in Heq. subst x. split; [ exact Hcanon_base | ].
        rewrite is_bound_cons, (Hne _ Hcanon_not),
          (Hnot_expanded _ Hcanon_not). reflexivity.
      * exact (Hremoved _ x d Hx).
    + destruct (Hdefs x d Hx) as [Hb Hn]. split; [ exact Hb | ].
      rewrite is_bound_cons, Hn.
      destruct (Pos.eqb x var) eqn:Hxv; [ | reflexivity ].
      apply Pos.eqb_eq in Hxv. subst x. rewrite Hvar_def in Hx.
      discriminate Hx.
  - destruct (unique var).
    + injection Hstep as <-.
      split; [ exact (Hexpand1 _) | exact (Hremoved _) ].
    + injection Hstep as <-. split; [ exact Hexpand | exact Hdefs ].
Qed.

Theorem last_phase_adds_definitions : last_phase_adds_definitions_statement.
Proof.
  intros tenv fuel source s relations kept equations relation_equations
    definitions Htrue Hnot_made Hadds x definition Hx.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept' [equations'
          [relation_equations' [Hfold [Hvars [_ [_ [_ [_ [_ Hr]]]]]]]]]]]]]].
  injection Hr as Hkept _ _. apply app_inv_head in Hkept. subst kept'.
  destruct (define_fold_kept (recording tenv) s tenv (unique_of s relations)
              _ _ _ Hfold) as [kept' [_ [Henv [_ Hall]]]].
  unfold acc_env in Henv. cbn [fst] in Henv.
  rewrite recording_defined in Henv. rewrite Henv in Hvars.
  cbn [recorded_variables] in Hvars. apply app_inv_head in Hvars.
  subst kept'.
  assert (Hinv :
            definitions_invariant s
              (r1, to_expand, equations1, relations1, definitions)).
  { refine (fold_left_invariant
              (define_step (recording tenv) s tenv (unique_of s relations))
              (definitions_invariant s) (fun b => eq_refl) _ _ _ _ _ Hfold).
    - intros a a' b Hb Ha Hstep.
      exact (define_step_definitions (recording tenv) source s tenv
               (unique_of s relations) a a' b Htrue Hnot_made Hb Ha Hstep).
    - split; [ intros e ty He; discriminate He | ].
      intros y d Hy. split; [ | reflexivity ].
      destruct Htrue as [[_ [Hnew [[Hrecorded _] _]]] _].
      destruct d as [simples k].
      pose proof (Hrecorded y simples k Hy) as He.
      unfold exists_in_target_env, kind_in_target_env.
      destruct (Keyed.lookup name_eqb (Name_var y)
                  (Typing_env.names_to_types
                     (Bindings_in_target_env.source_env
                        (Join_env.bindings s)))) as [[ty btm] | ];
        [ reflexivity | ].
      rewrite He. reflexivity. }
  destruct Hinv as [_ Hdefs]. destruct (Hdefs x definition Hx) as [Hb Hn].
  exact (base_in_final_target s to_expand kept Hall x Hb Hn).
Qed.
