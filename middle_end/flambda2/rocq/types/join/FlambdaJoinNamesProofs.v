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



(* The proofs of the statements of FlambdaJoinNamesSpec.v. This file
   needs no review.

   The relation is kept by every function of the join of environments
   that changes the state: only the one that makes a variable changes
   the names to avoid or the variables made. The lemmas follow those
   for the environments joined (FlambdaJoinStateProofs.v and
   FlambdaJoinDefineProofs.v, [same_envs]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinTypes FlambdaJoinTypesKeepsSpec
  FlambdaJoinTypesKeepsProofs FlambdaJoinNamesSpec.
Import ListNotations.

Lemma names_kept_refl (s : Join_env.t) : names_kept s s.
Proof. split; [ intros x Hx; exact Hx | intros x Hx; left; exact Hx ]. Qed.

Lemma names_kept_trans (s1 s2 s3 : Join_env.t) :
  names_kept s1 s2 -> names_kept s2 s3 -> names_kept s1 s3.
Proof.
  intros [Hnames12 Hmade12] [Hnames23 Hmade23]. split.
  - intros x Hx. exact (Hnames23 x (Hnames12 x Hx)).
  - intros x Hx. destruct (Hmade23 x Hx) as [H2 | H2].
    + exact (Hmade12 x H2).
    + right. intros H1. exact (H2 (Hnames12 x H1)).
Qed.

(* Two states with the same bindings. *)
Lemma same_bindings_names_kept (s s' : Join_env.t) :
  Join_env.bindings s' = Join_env.bindings s -> names_kept s s'.
Proof.
  intros Hsame. unfold names_kept, names_to_avoid, is_existential.
  rewrite Hsame. split; [ intros x Hx; exact Hx | intros x Hx; left; exact Hx ].
Qed.

Section Names.

Context (p : Join_env.parameters).

Hypothesis fresh : fresh_avoids p.

Lemma existential_names_kept (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  names_kept s s'.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0] eqn:Hbindings.
  injection Hmake as _ <-.
  unfold Bindings_in_target_env.existential_for_these_simples in Hbindings.
  destruct (Bindings_in_target_env.has_existential_for_these_simples
              (Join_env.bindings s) canonicals) as [existing | ].
  - injection Hbindings as _ <-. apply same_bindings_names_kept. reflexivity.
  - injection Hbindings as <- <-.
    unfold names_kept, names_to_avoid, is_existential.
    cbn [Join_env.bindings Bindings_in_target_env.names_to_avoid
           Bindings_in_target_env.existential_variables].
    split.
    + intros y Hy. right. exact Hy.
    + intros y Hy. unfold is_bound in Hy. cbn [lookup] in Hy.
      destruct (Pos.eqb y
                  (Join_env.fresh p
                     (Bindings_in_target_env.names_to_avoid
                        (Join_env.bindings s)))) eqn:Heq.
      * apply Pos.eqb_eq in Heq. subst y. right. apply fresh.
      * left. exact Hy.
Qed.

Lemma add_alias_names_kept (s : Join_env.t) (kind : flambda_kind)
    (var : variable) (canonicals : Simples_in_joined_envs.t) :
  names_kept s (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  exact (names_kept_trans _ _ _ (existential_names_kept _ _ _ _ _ Hmake)
           (same_bindings_names_kept _ _ eq_refl)).
Qed.

Lemma with_import_names_kept (s : Join_env.t) (x : variable)
    (kind : flambda_kind) :
  names_kept s (with_import s x kind).
Proof.
  unfold names_kept, names_to_avoid, is_existential, with_import.
  cbn [Join_env.bindings].
  unfold Bindings_in_target_env.import_from_all_envs.
  destruct (is_bound x
              (Bindings_in_target_env.imported_variables
                 (Join_env.bindings s)));
    cbn [Bindings_in_target_env.names_to_avoid
           Bindings_in_target_env.existential_variables];
    (split; [ intros y Hy; exact Hy | intros y Hy; left; exact Hy ]).
Qed.

Definition import_names_kept
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    import_var s x = Some s' -> names_kept s s'.

Lemma import_vars_names_kept
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_names_kept import_var ->
  ∀ s s' : Join_env.t,
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    names_kept s s'.
Proof.
  intros Himport.
  induction vars as [ | var vars IH]; intros s s' Hfold.
  - injection Hfold as <-. apply names_kept_refl.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + exact (names_kept_trans _ _ _ (Himport _ _ _ Hvar) (IH _ _ Hfold)).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equation_names_kept (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' -> names_kept s s'.
Proof.
  unfold Join_env.add_equation_in_target_env. intros Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-. apply same_bindings_names_kept. reflexivity.
Qed.

Lemma add_equals_names_kept
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  import_names_kept import_var ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  names_kept s s'.
Proof.
  intros Himport Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_names_kept _ _ _ _ Hadd).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    exact (names_kept_trans _ _ _ (Himport _ _ _ Hvar)
             (add_equation_names_kept _ _ _ _ Hadd)).
  - injection Hadd as <-. apply add_alias_names_kept.
Qed.

Lemma concrete_names_kept
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_names_kept import_var ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  names_kept s s'.
Proof.
  intros Himport Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    exact (names_kept_trans _ _ _ (add_equation_names_kept _ _ _ _ Hadd)
             (import_vars_names_kept _ _ Himport _ _ Hconcrete)).
  - injection Hconcrete as <-. apply same_bindings_names_kept. reflexivity.
Qed.

Lemma import_body_names_kept
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_names_kept import_var ->
  import_names_kept (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Himport s s' x Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x);
    [ injection Hbody as <-; apply names_kept_refl | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; apply names_kept_refl | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ]; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  apply (names_kept_trans _ _ _ (with_import_names_kept s x kind)).
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ];
    [ | discriminate ].
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ]; try discriminate.
    exact (add_equals_names_kept _ _ _ _ _ _ Himport Hbody).
  - exact (concrete_names_kept _ _ _ _ _ Himport Hbody).
Qed.

Lemma import_var_names_kept (fuel : nat) :
  import_names_kept (Join_env.import_var_transitive p fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' x H. discriminate.
  - exact (import_body_names_kept _ IH).
Qed.

Lemma import_or_delay_names_kept (fuel : nat) (s s' : Join_env.t)
    (var : variable) (types : Keyed.finite_map Index.t flambda_type) :
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  names_kept s s'.
Proof.
  intros Himport.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in Himport.
  destruct (Index.choose types) as [[index0 any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs s) types)
    as [[canonicals | ] | ]; [ | | discriminate ].
  - exact (add_equals_names_kept _ _ _ _ _ _ (import_var_names_kept fuel)
             Himport).
  - exact (concrete_names_kept _ _ _ _ _ (import_var_names_kept fuel)
             Himport).
Qed.

Lemma join_aliases_names_kept (fuel : nat) (envs : Joined_envs.t)
    (exists0 : variable -> bool) :
  ∀ (names : list name) (s s' : Join_env.t),
    List.fold_left
      (fun env name =>
         let* env := env in
         match name with
         | Name_symbol _ => Some env
         | Name_var var =>
           if exists0 var
           then
             Join_env.import_or_delay_n_way_join_type p fuel env var
               (Join_env.types_of_name_in_levels envs name)
           else Some env
         end)
      names (Some s)
    = Some s' ->
    names_kept s s'.
Proof.
  induction names as [ | n names IH]; intros s s' Hfold.
  - injection Hfold as <-. apply names_kept_refl.
  - cbn [List.fold_left] in Hfold.
    destruct n as [x | sym]; [ | exact (IH s s' Hfold) ].
    destruct (exists0 x); [ | exact (IH s s' Hfold) ].
    destruct (Join_env.import_or_delay_n_way_join_type p fuel s x
                (Join_env.types_of_name_in_levels envs (Name_var x)))
      as [s1 | ] eqn:Hs1.
    + exact (names_kept_trans _ _ _
               (import_or_delay_names_kept _ _ _ _ _ Hs1) (IH s1 s' Hfold)).
    + exfalso. clear -Hfold.
      induction names as [ | n' names IH]; [ discriminate | exact (IH Hfold) ].
Qed.

(* The three functions that the join of types calls. *)

Lemma import_type_names_kept (fuel : nat) (s s' : Join_env.t)
    (ty : flambda_type) :
  Join_env.import_type p fuel s ty = Some s' -> names_kept s s'.
Proof.
  intros Himport.
  exact (import_vars_names_kept _ (free_variables ty)
           (import_var_names_kept fuel) s s' Himport).
Qed.

Lemma n_way_join_simples_names_kept (fuel : nat) (s s' : Join_env.t)
    (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple) :
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  names_kept s s'.
Proof.
  intros Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. apply names_kept_refl. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. apply names_kept_refl.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_names_kept fuel _ _ _ Himport).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_names_kept _ _ _ _ _ Hmake).
Qed.

Lemma n_way_join_env_extension_names_kept (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension) :
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  names_kept s s'.
Proof.
  intros Hjoin.
  unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply names_kept_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : names_kept s s2).
        { destruct n as [x | sym].
          - exact (import_var_names_kept fuel _ _ _ Hname).
          - injection Hname as <-. apply names_kept_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (names_kept_trans _ _ _ Hfirst
                    (names_kept_trans _ _ _
                       (import_type_names_kept fuel _ _ _ Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply names_kept_refl.
Qed.

End Names.

Theorem n_way_join_keeps_names : n_way_join_keeps_names_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb Hfresh s s' ts joined Hstep.
  refine
    (n_way_join_keeps Join_env.t names_kept Join_env.joined_env
       (Join_env.n_way_join_simples p fuel_of_env)
       (Join_env.import_type p fuel_of_env)
       (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
       names_kept_refl names_kept_trans _ _ _ s s' ts joined Hstep).
  - intros s0 s0' kind simples result Hsimples.
    exact (n_way_join_simples_names_kept p Hfresh fuel_of_env s0 s0' kind
             simples result Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_names_kept p Hfresh fuel_of_env s0 s0' ty Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact (n_way_join_env_extension_names_kept p Hfresh fuel_of_env s0 s0'
             extensions joined0 Hextension).
Qed.

Section Rounds_names.

Context (jt : Join_env.n_way_join_type) (choices : Join_env.choices).

Hypothesis jt_keeps : join_of_types_keeps_names jt.

Lemma round_names_kept (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  names_kept s s'.
Proof.
  intros Hround. unfold Join_env.n_way_join_round in Hround.
  refine (fold_left_invariant _
            (fun a : Join_env.inverse_relations * Join_env.t =>
               names_kept s (snd a))
            (fun b => eq_refl) _ _ (relations, s) (relations', s')
            (names_kept_refl s) Hround).
  intros [relations0 s0] [relations1 s1] [var types] _ H0 Hstep.
  cbn [fst snd] in Hstep, H0 |- *.
  destruct (jt s0 (Index.bindings types)) as [[joined s2] | ] eqn:Hjoin;
    [ | discriminate ].
  pose proof (names_kept_trans _ _ _ H0 (jt_keeps _ _ _ _ Hjoin)) as H2.
  destruct joined as [ty | ].
  - destruct (Join_env.recover_relations_on choices s2 var types).
    + destruct (Join_env.recover_inverse_relations relations0 (Name_var var)
                  ty) as [[ty' relations2] | ]; [ | discriminate ].
      destruct (Join_env.add_equation_in_target_env s2 var ty') as [s3 | ]
        eqn:Hadd; [ | discriminate ].
      injection Hstep as _ <-.
      exact (names_kept_trans _ _ _ H2
               (add_equation_names_kept _ _ _ _ Hadd)).
    + destruct (Join_env.add_equation_in_target_env s2 var ty) as [s3 | ]
        eqn:Hadd; [ | discriminate ].
      injection Hstep as _ <-.
      exact (names_kept_trans _ _ _ H2
               (add_equation_names_kept _ _ _ _ Hadd)).
  - injection Hstep as _ <-. exact H2.
Qed.

Lemma delayed_equations_names_kept :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type)),
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    names_kept s s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' relations relations' equations Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. apply names_kept_refl.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    exact (names_kept_trans _ _ _ (round_names_kept _ _ _ _ _ Hround)
             (IH s1 s' relations1 relations' _ Hloop)).
Qed.

Lemma loop_names_kept (p : Join_env.parameters) :
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    names_kept s s'.
Proof.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. apply names_kept_refl.
  - destruct (Keyed.is_empty existentials).
    + injection Hloop as _ <-. apply names_kept_refl.
    + destruct (Join_env.equations_of_existentials p s (compact existentials))
        as [equations | ]; [ | discriminate ].
      destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                  relations equations) as [[relations1 s1] | ] eqn:Hdelayed;
        [ | discriminate ].
      exact (names_kept_trans _ _ _
               (delayed_equations_names_kept _ _ _ _ _ _ Hdelayed)
               (IH fuel s1 s' relations1 relations' _ Hloop)).
Qed.

End Rounds_names.

Theorem first_phases_names : first_phases_names_statement.
Proof.
  intros jt choices p fuel avoid source envs s1 s2 s3 relations2 relations3
    Hjt Hfresh Haliases Hdelayed Hloop.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  unfold Join_env.join_aliases_into_bindings in Haliases.
  pose proof (join_aliases_names_kept p Hfresh fuel envs _ _ s0 s1 Haliases)
    as H1.
  pose proof (delayed_equations_names_kept jt choices Hjt _ _ _ _ _ _
                Hdelayed) as H2.
  pose proof (loop_names_kept jt choices Hjt p _ _ _ _ _ _ _ Hloop) as H3.
  destruct (names_kept_trans _ _ _ (names_kept_trans _ _ _ H1 H2) H3)
    as [Hnames Hmade].
  split.
  - intros x Hx. exact (Hnames x Hx).
  - intros x Hx. destruct (Hmade x Hx) as [H | H]; [ discriminate H | ].
    exact H.
Qed.
