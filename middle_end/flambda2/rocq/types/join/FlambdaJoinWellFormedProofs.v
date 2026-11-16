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


(* The proofs of the statements of FlambdaJoinWellFormedSpec.v. This
   file needs no review.

   Each function of the join of environments keeps
   [equations_well_formed] where the joined environments have
   well-formed types: they are shown in the order of
   FlambdaJoinEnv.v, as for the definitions (FlambdaJoinEnvProofs.v).
   Then the rounds, which follow those of FlambdaJoinStateProofs.v,
   and the last phase. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeProject FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvExtension FlambdaTypingEnvExtensionSpec
  FlambdaTypingEnvExtensionProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinFitsSpec FlambdaJoinFitsProofs
  FlambdaJoinAddsSpec FlambdaJoinAddsProofs FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesKeepsSpec FlambdaJoinTypesKeepsProofs
  FlambdaJoinTypesWithEnvSpec FlambdaJoinMeet FlambdaJoinWellFormedSpec.
From OxCamlFlambda2Types Require FlambdaExpandHead FlambdaExpandHeadSpec
  FlambdaExpandHeadProofs.
Import ListNotations.

Abbreviation envs_well_formed := (joined_envs_are types_well_formed).

(* ======================================================================== *)
(* The functions of the join of environments                                *)
(* ======================================================================== *)

Lemma keeps_envs_well_formed (s s' : Join_env.t) :
  keeps_definitions s s' -> envs_well_formed s -> envs_well_formed s'.
Proof.
  intros [Henvs _] H index use Huse. rewrite Henvs in Huse.
  exact (H _ _ Huse).
Qed.

(* The same types: all is kept. *)
Lemma well_formed_same_types (s s' : Join_env.t) :
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.types_in_joined_envs s' = Join_env.types_in_joined_envs s ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  intros Htarget Hjoined [H1 H2]. split.
  - intros x ty Hx. rewrite Htarget in Hx. exact (H1 x ty Hx).
  - intros x types index t Hx Ht. rewrite Hjoined in Hx.
    exact (H2 x types index t Hx Ht).
Qed.

Lemma existential_keeps_well_formed (p : Join_env.parameters)
    (s s' : Join_env.t) (canonicals : Simples_in_joined_envs.t)
    (kind : flambda_kind) (e : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (e, s') ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0].
  injection Hmake as _ <-. apply well_formed_same_types; reflexivity.
Qed.

Lemma add_equation_keeps_well_formed (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  type_well_formed ty = true ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  intros Hadd Hty [H1 H2].
  unfold Join_env.add_equation_in_target_env in Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-. split; [ | exact H2 ].
  intros x0 ty0 Hx0. cbn [Join_env.types_in_target_env] in Hx0.
  rewrite lookup_cons in Hx0. destruct (Pos.eqb x0 x).
  - injection Hx0 as <-. exact Hty.
  - exact (H1 x0 ty0 Hx0).
Qed.

Lemma add_alias_keeps_well_formed (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  equations_well_formed s ->
  equations_well_formed
    (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros H. unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  pose proof (existential_keeps_well_formed _ _ _ _ _ _ Hmake H) as H1.
  revert H1. apply well_formed_same_types; reflexivity.
Qed.

(* What is shown of a function that imports a variable. *)
Definition import_keeps_well_formed
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    envs_well_formed s ->
    import_var s x = Some s' ->
    equations_well_formed s ->
    equations_well_formed s'.

Lemma import_vars_keeps_well_formed
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps_definitions import_var ->
  import_keeps_well_formed import_var ->
  ∀ s s' : Join_env.t,
    envs_well_formed s ->
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    equations_well_formed s ->
    equations_well_formed s'.
Proof.
  intros Hdefinitions Himport.
  induction vars as [ | var vars IH]; intros s s' Henvs Hfold H.
  - injection Hfold as <-. exact H.
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + exact (IH s1 s'
               (keeps_envs_well_formed _ _ (Hdefinitions _ _ _ Hvar) Henvs)
               Hfold (Himport _ _ _ Henvs Hvar H)).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equals_keeps_well_formed (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  import_keeps_well_formed import_var ->
  envs_well_formed s ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  intros Himport Henvs Hadd H.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ]; [ | | | discriminate ].
  - exact (add_equation_keeps_well_formed _ _ _ _ Hadd
             (alias_type_of_well_formed kind canonical) H).
  - destruct (import_var s imported) as [s1 | ] eqn:Hvar; [ | discriminate ].
    exact (add_equation_keeps_well_formed _ _ _ _ Hadd
             (alias_type_of_well_formed kind (Var imported))
             (Himport _ _ _ Henvs Hvar H)).
  - injection Hadd as <-. apply add_alias_keeps_well_formed. exact H.
Qed.

Lemma concrete_keeps_well_formed (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps_definitions import_var ->
  import_keeps_well_formed import_var ->
  envs_well_formed s ->
  (∀ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     type_well_formed t = true) ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  intros Hdefinitions Himport Henvs Htypes Hconcrete H.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  pose proof (choose_lookup _ _ _ Hchoose) as Hany.
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    exact (import_vars_keeps_well_formed import_var (free_variables any_ty)
             Hdefinitions Himport s1 s'
             (keeps_envs_well_formed _ _
                (add_equation_keeps_definitions _ _ _ _ Hadd) Henvs)
             Hconcrete
             (add_equation_keeps_well_formed _ _ _ _ Hadd
                (Htypes _ _ Hany) H)).
  - injection Hconcrete as <-. destruct H as [H1 H2]. split; [ exact H1 | ].
    intros x types0 index0 t Hx Ht.
    cbn [Join_env.add_equations_in_joined_envs Join_env.types_in_joined_envs]
      in Hx.
    rewrite lookup_cons in Hx. destruct (Pos.eqb x var).
    + injection Hx as <-. exact (Htypes _ _ Ht).
    + exact (H2 x types0 index0 t Hx Ht).
Qed.

(* The types that the joined environments have for a variable brought
   in are well formed. *)
Lemma imported_types_well_formed (s : Join_env.t) (x : variable)
    (kind : flambda_kind) (types : Keyed.finite_map Index.t flambda_type) :
  envs_well_formed s ->
  Joined_envs.find_imported_var (Join_env.joined_envs s) x kind = Some types
  ->
  ∀ (index : Index.t) (t : flambda_type),
    Keyed.lookup Index.equal index types = Some t ->
    type_well_formed t = true.
Proof.
  intros Henvs Htypes index t Ht.
  destruct (types_of_imported_var_spec _ _ _ _ Htypes) as [_ Hfound].
  apply (keyed_lookup_in Index.equal index_equal_decides) in Ht.
  destruct (Hfound _ _ Ht) as [env [equations [Hin Hfind]]].
  pose proof (in_compact_joined_env _ _ _ _ Hin) as Huse.
  exact (find_well_formed env (Name_var x) (Some kind) t (Henvs _ _ Huse)
           Hfind).
Qed.

Lemma import_body_keeps_well_formed (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_keeps_definitions import_var ->
  import_keeps_well_formed import_var ->
  import_keeps_well_formed (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Hdefinitions Himport s s' x Henvs Hbody H.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x);
    [ injection Hbody as <-; exact H | ].
  destruct (among x (Join_env.imported_vars s));
    [ injection Hbody as <-; exact H | ].
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ]; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  assert (Henvs1 : envs_well_formed (with_import s x kind)) by exact Henvs.
  assert (H1 : equations_well_formed (with_import s x kind)).
  { revert H. apply well_formed_same_types; reflexivity. }
  destruct (Joined_envs.find_imported_var _ x kind) as [types | ] eqn:Htypes;
    [ | discriminate ].
  destruct (existsb _ _).
  - destruct
      (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
         _ _) as [[canonical_simples | ] | ]; try discriminate.
    exact (add_equals_keeps_well_formed _ _ _ _ _ _ _ Himport Henvs1 Hbody
             H1).
  - exact (concrete_keeps_well_formed _ _ _ _ _ _ Hdefinitions Himport Henvs1
             (imported_types_well_formed s x kind types Henvs Htypes) Hbody
             H1).
Qed.

Lemma import_var_keeps_well_formed (p : Join_env.parameters) (fuel : nat) :
  import_keeps_well_formed (Join_env.import_var_transitive p fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' x _ H. discriminate.
  - exact (import_body_keeps_well_formed p _
             (import_var_keeps_definitions p fuel) IH).
Qed.

(* ======================================================================== *)
(* The join of types                                                        *)
(* ======================================================================== *)

(* What every function of the join of environments keeps, between any
   two states. *)
Definition keeps_well_formed (s s' : Join_env.t) : Prop :=
  keeps_definitions s s'
  ∧ (envs_well_formed s -> equations_well_formed s -> equations_well_formed s').

Lemma keeps_well_formed_refl (s : Join_env.t) : keeps_well_formed s s.
Proof.
  split; [ apply keeps_definitions_refl | intros _ H; exact H ].
Qed.

Lemma keeps_well_formed_trans (s1 s2 s3 : Join_env.t) :
  keeps_well_formed s1 s2 -> keeps_well_formed s2 s3 ->
  keeps_well_formed s1 s3.
Proof.
  intros [D12 K12] [D23 K23].
  split; [ exact (keeps_definitions_trans _ _ _ D12 D23) | ].
  intros Henvs H.
  exact (K23 (keeps_envs_well_formed _ _ D12 Henvs) (K12 Henvs H)).
Qed.

Lemma import_var_keeps_well_formed_between (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t) (x : variable) :
  Join_env.import_var_transitive p fuel s x = Some s' ->
  keeps_well_formed s s'.
Proof.
  intros Hi. split.
  - exact (import_var_keeps_definitions p fuel _ _ _ Hi).
  - intros Henvs H.
    exact (import_var_keeps_well_formed p fuel _ _ _ Henvs Hi H).
Qed.

Lemma import_type_keeps_well_formed_between (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t) (ty : flambda_type) :
  Join_env.import_type_transitive p fuel s ty = Some s' ->
  keeps_well_formed s s'.
Proof.
  intros Hi. split.
  - exact (import_type_keeps_definitions p fuel s s' ty Hi).
  - intros Henvs H.
    exact (import_vars_keeps_well_formed _ (free_variables ty)
             (import_var_keeps_definitions p fuel)
             (import_var_keeps_well_formed p fuel) s s' Henvs Hi H).
Qed.

Lemma n_way_join_simples_keeps_well_formed (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) (result : Or_bottom.t simple) :
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  keeps_well_formed s s'.
Proof.
  intros Hjoin. split.
  { exact (n_way_join_simples_keeps_definitions p fuel s s' kind simples
             result Hjoin). }
  intros Henvs H. unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. exact H. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. exact H.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_keeps_well_formed p fuel _ _ _ Henvs Himport H).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_keeps_well_formed _ _ _ _ _ _ Hmake H).
Qed.

Lemma n_way_join_env_extension_keeps_well_formed (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension) :
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  keeps_well_formed s s'.
Proof.
  intros Hjoin. unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply keeps_well_formed_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : keeps_well_formed s s2).
        { destruct n as [x | sym].
          - exact (import_var_keeps_well_formed_between p fuel _ _ _ Hname).
          - injection Hname as <-. apply keeps_well_formed_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (keeps_well_formed_trans _ _ _ Hfirst
                    (keeps_well_formed_trans _ _ _
                       (import_type_keeps_well_formed_between p fuel _ _ _
                          Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply keeps_well_formed_refl.
Qed.

Theorem n_way_join_keeps_well_formed : n_way_join_keeps_well_formed_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb s s' ts joined Hstep Henvs H.
  refine
    (proj2
       (n_way_join_keeps Join_env.t keeps_well_formed Join_env.joined_env
          (Join_env.n_way_join_simples p fuel_of_env)
          (Join_env.import_type p fuel_of_env)
          (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
          reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
          keeps_well_formed_refl keeps_well_formed_trans _ _ _ s s' ts joined
          Hstep)
       Henvs H).
  - intros s0 s0' kind simples result Hsimples.
    exact (n_way_join_simples_keeps_well_formed p fuel_of_env s0 s0' kind
             simples result Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_keeps_well_formed_between p fuel_of_env s0 s0' ty
             Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact (n_way_join_env_extension_keeps_well_formed p fuel_of_env s0 s0'
             extensions joined0 Hextension).
Qed.

(* ======================================================================== *)
(* The relations recovered                                                  *)
(* ======================================================================== *)

Lemma with_inverse_relation_well_formed (head : head_of_kind_naked_immediate)
    (relation : Relation.t) (scrutinee : name) :
  head_of_kind_naked_immediate_well_formed head = true ->
  type_well_formed (Join_env.with_inverse_relation head relation scrutinee)
  = true.
Proof. destruct head as [numbers relations]. intros H. exact H. Qed.

Lemma add_or_replace_well_formed (extension : env_extension) (n : name)
    (ty : flambda_type) :
  env_extension_well_formed extension = true ->
  type_well_formed ty = true ->
  env_extension_well_formed (Join_env.add_or_replace_equation extension n ty)
  = true.
Proof.
  destruct extension as [equations]. intros Hextension Hty.
  apply env_extension_well_formed_iff in Hextension.
  destruct Hextension as [Hdistinct Hall].
  unfold Join_env.add_or_replace_equation.
  apply env_extension_well_formed_iff. split.
  - exact (add_or_replace_equation_distinct (Env_extension equations) n ty
             Hdistinct).
  - intros n' ty' Hlookup. rewrite keyed_lookup_cons in Hlookup.
    destruct (name_eqb n' n) eqn:Heq.
    + injection Hlookup as <-. exact Hty.
    + rewrite (keyed_lookup_unbind name name_eqb name_eqb_eq), Heq in Hlookup.
      exact (Hall n' ty' Hlookup).
Qed.

Theorem recover_inverse_relations_well_formed
  : recover_inverse_relations_well_formed_statement.
Proof.
  intros relations relations' n ty ty' Hrecover Hty.
  destruct (recover_type_cases _ _ _ _ _ Hrecover)
    as [-> | [is_int [immediates [g [blocks [extensions [is_unique
          [when_block [-> [Hadd ->]]]]]]]]]]; [ exact Hty | ].
  unfold variant_type in Hty |- *.
  cbn [type_well_formed descr_well_formed head_of_kind_value_well_formed
         head_of_kind_value_non_null_well_formed] in Hty |- *.
  apply andb_true_iff in Hty. destruct Hty as [Hty Hextensions].
  rewrite Hty. cbn [andb].
  assert (Hpair :
            env_extension_well_formed (fst (extensions_pair extensions)) = true
            ∧ env_extension_well_formed (snd (extensions_pair extensions))
              = true).
  { destruct extensions as [ | a b]; cbn [extensions_pair fst snd].
    - split; reflexivity.
    - apply andb_true_iff in Hextensions. exact Hextensions. }
  destruct Hpair as [Hfst Hsnd]. rewrite Hfst. cbn [andb].
  destruct (add_inverse_relation_spec _ _ _ _ _ _ _ Hadd)
    as [-> | [n' [head [_ [Hhead ->]]]]]; [ exact Hsnd | ].
  apply add_or_replace_well_formed; [ exact Hsnd | ].
  apply with_inverse_relation_well_formed.
  destruct Hhead as [-> | Hlookup]; [ reflexivity | ].
  destruct (snd (extensions_pair extensions)) as [equations].
  apply env_extension_well_formed_iff in Hsnd.
  exact (proj2 Hsnd n' _ Hlookup).
Qed.

(* ======================================================================== *)
(* The first phase                                                          *)
(* ======================================================================== *)

(* Types that are all well formed. *)
Definition all_well_formed (types : Keyed.finite_map Index.t flambda_type)
    : Prop :=
  ∀ (index : Index.t) (t : flambda_type),
    Keyed.lookup Index.equal index types = Some t ->
    type_well_formed t = true.

Lemma import_or_delay_keeps_definitions (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  keeps_definitions s s'.
Proof.
  intros H.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ]; [ | | discriminate ].
  - exact (add_equals_keeps_definitions _ _ _ _ _ _ _
             (import_var_keeps_definitions p fuel) H).
  - exact (concrete_keeps_definitions _ _ _ _ _ _
             (import_var_keeps_definitions p fuel) H).
Qed.

Lemma import_or_delay_keeps_well_formed (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  envs_well_formed s ->
  all_well_formed types ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  equations_well_formed s ->
  equations_well_formed s'.
Proof.
  intros Henvs Htypes H HX.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ]; [ | | discriminate ].
  - exact (add_equals_keeps_well_formed p _ s s' _ var canonical_simples
             (import_var_keeps_well_formed p fuel) Henvs H HX).
  - exact (concrete_keeps_well_formed p _ s s' var types
             (import_var_keeps_definitions p fuel)
             (import_var_keeps_well_formed p fuel) Henvs Htypes H HX).
Qed.

(* The step of the fold of the first phase. *)
Definition aliases_fold_step (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool)
    (env : option Join_env.t) (name : name) : option Join_env.t :=
  let* env := env in
  match name with
  | Name_symbol _ => Some env
  | Name_var var =>
    if test var
    then
      Join_env.import_or_delay_n_way_join_type p fuel env var
        (Join_env.types_of_name_in_levels envs name)
    else Some env
  end.

Lemma aliases_fold_keeps_well_formed (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool) :
  (∀ x : variable,
     all_well_formed (Join_env.types_of_name_in_levels envs (Name_var x))) ->
  ∀ (names : list name) (s s' : Join_env.t),
    envs_well_formed s ->
    List.fold_left (aliases_fold_step p fuel envs test) names (Some s)
    = Some s' ->
    equations_well_formed s ->
    equations_well_formed s'.
Proof.
  intros Htypes.
  induction names as [ | n names IH]; intros s s' Henvs Hfold HX.
  - injection Hfold as <-. exact HX.
  - cbn [List.fold_left] in Hfold.
    destruct (aliases_fold_step p fuel envs test (Some s) n) as [s1 | ]
      eqn:Hstep;
      [ | rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold;
          discriminate ].
    assert (Hfirst : keeps_definitions s s1 ∧ equations_well_formed s1).
    { destruct n as [var | sym]; cbn [aliases_fold_step] in Hstep.
      - destruct (test var).
        + split.
          * exact (import_or_delay_keeps_definitions _ _ _ _ _ _ Hstep).
          * exact (import_or_delay_keeps_well_formed p fuel s s1 var _ Henvs
                     (Htypes var) Hstep HX).
        + injection Hstep as <-.
          split; [ apply keeps_definitions_refl | exact HX ].
      - injection Hstep as <-.
        split; [ apply keeps_definitions_refl | exact HX ]. }
    destruct Hfirst as [Hkeeps HX1].
    exact (IH s1 s' (keeps_envs_well_formed _ _ Hkeeps Henvs) Hfold HX1).
Qed.

(* The types that the joined environments have for the [Simple]s of a
   vector are well formed. *)
Lemma simples_types_well_formed (p : Join_env.parameters) (s : Join_env.t)
    (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (types : Keyed.finite_map Index.t flambda_type) :
  expand_const_well_formed p ->
  envs_well_formed s ->
  Joined_envs.find_simples_in_joined_envs (Join_env.expand_const p)
    (Join_env.joined_envs s) simples kind
  = Some types ->
  all_well_formed types.
Proof.
  intros Hconst Henvs Hfind index t Ht.
  unfold Joined_envs.find_simples_in_joined_envs in Hfind.
  destruct (types_of_simples_lookup _ _ _ _ _ Hfind) as [Hfound _].
  destruct (Hfound index t Ht) as [simple [_ Hty]].
  destruct simple as [y | sym | c]; cbn [type_of_simple_at] in Hty.
  - destruct (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
      as [use | ] eqn:Huse; [ | discriminate ].
    cbn in Hty.
    exact (find_well_formed use _ _ t (Henvs index use Huse) Hty).
  - destruct (Joined_envs.get_nth_joined_env (Join_env.joined_envs s) index)
      as [use | ] eqn:Huse; [ | discriminate ].
    cbn in Hty.
    exact (find_well_formed use _ _ t (Henvs index use Huse) Hty).
  - injection Hty as <-. apply Hconst.
Qed.

(* ======================================================================== *)
(* The rounds                                                               *)
(* ======================================================================== *)

Lemma later_envs_well_formed (s s' : Join_env.t) :
  later s s' -> envs_well_formed s -> envs_well_formed s'.
Proof.
  intros Hlater H index use Huse. rewrite (later_env _ _ _ Hlater) in Huse.
  exact (H _ _ Huse).
Qed.

Section Rounds_well_formed.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

Hypothesis jt_well_formed :
  n_way_join_type_well_formed typing_env_invariant Join_env.joined_env
    state_valid state_kinded jt.

Hypothesis jt_keeps : join_of_types_keeps_well_formed jt.

Lemma join_call_well_formed (s s0 : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) (ty : flambda_type) :
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  envs_well_formed s ->
  types_hold s var types ->
  all_well_formed types ->
  state_kinded s ->
  jt s (Index.bindings types) = Some (Or_unknown.Known ty, s0) ->
  type_well_formed ty = true.
Proof.
  intros Htrue Hinvariant Hkinded Hwf [Hvar [Hfound _]] Htypes Hstate Hjoin.
  apply (jt_well_formed s s0 (Index.bindings types) ty (proj1 Htrue)
           Hinvariant Hkinded Hwf (bindings_distinct types));
    [ | | | exact Hstate | exact Hjoin ].
  - intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [Hscope _]]].
    exists use. split; [ exact Huse | exact Hscope ].
  - intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [_ Hkind]]].
    exists use. split; [ exact Huse | exact Hkind ].
  - intros index t Hin. apply bindings_lookup in Hin.
    exact (Htypes _ _ Hin).
Qed.

Lemma round_fold_well_formed :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings ->
       types_hold s var types ∧ all_well_formed types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_well_formed s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_well_formed s ->
    List.fold_left (round_step jt choices) bindings (Some (relations, s))
    = Some (relations', s') ->
    equations_well_formed s'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded Hwf
      Hrelations Hstate HX Hfold.
  - cbn in Hfold. injection Hfold as _ <-. exact HX.
  - cbn [List.fold_left] in Hfold.
    destruct (round_step jt choices (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    destruct (Htypes var types (or_introl eq_refl))
      as [Hvar_types Hvar_wf].
    assert (Hsingle :
              ∀ (v : variable) (t : Keyed.finite_map Index.t flambda_type),
                In (v, t) [(var, types)] -> types_hold s v t).
    { intros v t [Heq | []]. injection Heq as <- <-. exact Hvar_types. }
    destruct (round_fold_holds source jt choices jt_sound [(var, types)] s s1
                relations relations1 Hsingle Htrue Hinvariant Hkinded
                Hrelations Hstep)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (round_fold_kinded source jt choices jt_sound [(var, types)] s
                  s1 relations relations1 Hsingle Htrue Hinvariant Hkinded
                  Hrelations Hstate Hstep) as Hstate1.
    assert (HX1 : equations_well_formed s1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      pose proof (jt_keeps s s0 _ joined Hjoin Hwf HX) as HX0.
      destruct joined as [ty | ].
      - pose proof (join_call_well_formed s s0 var types ty Htrue Hinvariant
                      Hkinded Hwf Hvar_types Hvar_wf Hstate Hjoin) as Hty.
        destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-.
          exact (add_equation_keeps_well_formed _ _ _ _ Hadd2
                   (recover_inverse_relations_well_formed _ _ _ _ _ Hrecover
                      Hty)
                   HX0).
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-.
          exact (add_equation_keeps_well_formed _ _ _ _ Hadd2 Hty HX0).
      - injection Hstep as _ <-. exact HX0. }
    apply (IH s1 s' relations1 relations');
      [ | exact (later_true source s s1 Hlater1 Hholds1
                   (proj2 (proj2 Htrue)))
        | exact (later_envs_invariant _ _ Hlater1 Hinvariant)
        | exact (later_envs_kinded _ _ Hlater1 Hkinded)
        | exact (later_envs_well_formed _ _ Hlater1 Hwf)
        | exact Hrelations1 | exact Hstate1 | exact HX1 | exact Hfold ].
    intros v t Hin. destruct (Htypes v t (or_intror Hin)) as [Hh Hk].
    split; [ exact (later_types_hold _ _ _ _ Hlater1 Hh) | exact Hk ].
Qed.

Lemma delayed_equations_well_formed :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types ->
       types_hold s var types ∧ all_well_formed types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_well_formed s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_well_formed s ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    equations_well_formed s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Hwf Hrelations Hstate HX Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. exact HX.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    assert (Hholds_types :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar. exact (proj1 (Htypes var types Hvar)). }
    destruct (round_holds source jt choices jt_sound s s1 equations relations
                relations1 Hholds_types Htrue Hinvariant Hkinded Hrelations
                Hround)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (round_fold_kinded source jt choices jt_sound
                  (compact equations) s s1 relations relations1
                  (fun var types Hin =>
                     Hholds_types var types (proj1 (in_compact _ _ _) Hin))
                  Htrue Hinvariant Hkinded Hrelations Hstate Hround)
      as Hstate1.
    assert (HX1 : equations_well_formed s1).
    { apply (round_fold_well_formed (compact equations) s s1 relations
               relations1);
        [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Hwf
          | exact Hrelations | exact Hstate | exact HX | exact Hround ].
      intros var types Hin. apply Htypes. apply in_compact. exact Hin. }
    assert (Htypes1 :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var (Join_env.new_equations_in_joined_envs s1 s)
                = Some types ->
                types_hold s1 var types ∧ all_well_formed types).
    { intros var types Hvar.
      unfold Join_env.new_equations_in_joined_envs in Hvar.
      rewrite lookup_filter_bindings in Hvar.
      destruct (lookup var (Join_env.types_in_joined_envs s1))
        as [types1 | ] eqn:Hlookup; [ | discriminate ].
      destruct (negb (is_bound var (Join_env.types_in_joined_envs s)));
        [ | discriminate ].
      injection Hvar as <-.
      split; [ exact (proj1 (proj2 Hholds1) var types1 Hlookup) | ].
      intros index t Ht. exact (proj2 HX1 var types1 index t Hlookup Ht). }
    exact (IH s1 s' _ relations1 relations' Htypes1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded)
             (later_envs_well_formed _ _ Hlater1 Hwf) Hrelations1 Hstate1 HX1
             Hloop).
Qed.

Lemma loop_well_formed (p : Join_env.parameters) :
  expand_const_sound p ->
  expand_const_well_formed p ->
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    envs_well_formed s ->
    relations_hold s relations ->
    state_kinded s ->
    equations_well_formed s ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    equations_well_formed s'.
Proof.
  intros Hconst Hconst_wf.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Hwf Hrelations Hstate HX Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. exact HX.
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as _ <-. exact HX. }
    destruct (Join_env.equations_of_existentials p s (compact existentials))
      as [equations | ] eqn:Hequations; [ | discriminate ].
    destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                relations equations)
      as [[relations1 s1] | ] eqn:Hdelayed; [ | discriminate ].
    assert (Htypes :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types ->
                types_hold s var types ∧ all_well_formed types).
    { intros var types Hvar.
      destruct (equations_of_existentials_lookup p s _ _ Hequations _ _ Hvar)
        as [simples [kind [Hin Hfind]]].
      apply in_compact in Hin. split.
      - exact (existential_types_hold p s var simples kind types Hconst
                 (proj1 Htrue) (proj1 (proj2 Htrue)) Hinvariant Hkinded
                 (Hdefinitions _ _ Hin) Hfind).
      - exact (simples_types_well_formed p s simples kind types Hconst_wf Hwf
                 Hfind). }
    assert (Hholds_types :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar. exact (proj1 (Htypes var types Hvar)). }
    destruct (delayed_equations_hold source jt choices jt_sound fuel s s1
                equations relations relations1 Hholds_types Htrue Hinvariant
                Hkinded Hrelations Hdelayed)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (delayed_equations_kinded source jt choices jt_sound fuel s s1
                  equations relations relations1 Hholds_types Htrue Hinvariant
                  Hkinded Hrelations Hstate Hdelayed) as Hstate1.
    pose proof (delayed_equations_well_formed fuel s s1 equations relations
                  relations1 Htypes Htrue Hinvariant Hkinded Hwf Hrelations
                  Hstate HX Hdelayed) as HX1.
    assert (Hdefinitions1 :
              ∀ (x : variable)
                (definition : Simples_in_joined_envs.t * flambda_kind),
                lookup x (Join_env.new_definitions_of_existentials s1 s)
                = Some definition ->
                lookup x (Join_env.definitions_of_existentials s1)
                = Some definition).
    { intros x definition Hx.
      unfold Join_env.new_definitions_of_existentials in Hx.
      rewrite lookup_filter_bindings in Hx.
      destruct (lookup x (Join_env.definitions_of_existentials s1))
        as [definition1 | ]; [ | discriminate ].
      destruct (negb (is_bound x (Join_env.definitions_of_existentials s)));
        [ | discriminate ].
      exact Hx. }
    exact (IH fuel s1 s' relations1 relations' _ Hdefinitions1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded)
             (later_envs_well_formed _ _ Hlater1 Hwf) Hrelations1 Hstate1 HX1
             Hloop).
Qed.

End Rounds_well_formed.

Theorem first_phases_well_formed : first_phases_well_formed_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hjt_wf Hkeeps Hfresh Heqb Hconst Hconst_wf Hsource Huses Hinvariant
    Hkinded Hwf Hlevels Hkinds Haliases Hdelayed Hloop.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  pose proof (created_state_true avoid source envs Hsource Huses) as Htrue0.
  fold s0 in Htrue0.
  pose proof (created_state_kinded avoid source envs Hkinds) as Hstate0.
  fold s0 in Hstate0.
  assert (Hknown0 :
            ∀ x : variable, exists_in_target s0 x = true -> knows source x).
  { intros x Hx. rewrite kind_in_target_env_exists in Hx.
    unfold s0, Join_env.create, Bindings_in_target_env.exists_in_target_env,
      Bindings_in_target_env.from_source_env in Hx.
    cbn in Hx. rewrite !orb_false_r in Hx. exact Hx. }
  assert (Hinvariant0 : envs_invariant s0) by exact Hinvariant.
  assert (Hkinded0 : envs_kinded s0) by exact Hkinded.
  assert (Hwf0 : envs_well_formed s0) by exact Hwf.
  destruct (join_aliases_true p fuel source s0 s1 Hfresh Heqb Htrue0 Hstate0
              Hinvariant Hkinded Hlevels Hknown0 Haliases)
    as [Htrue1 Hgrows1].
  pose proof (join_aliases_kinded p fuel s0 s1 Hfresh Heqb (proj1 Htrue0)
                Hstate0 Hinvariant0 Hkinded0 (proj1 (proj2 Htrue0)) Hlevels
                Hknown0 Haliases) as Hstate1.
  assert (Hlater1 : later s0 s1).
  { split; [ exact (proj1 Htrue0) | ]. split; [ exact (proj1 Htrue1) | ].
    split; [ exact Hgrows1 | ].
    rewrite (proj2 (proj2 Htrue1)). symmetry. exact (proj2 (proj2 Htrue0)). }
  assert (HX0 : equations_well_formed s0).
  { split; [ intros x ty Hx; discriminate | ].
    intros x types index t Hx. discriminate. }
  assert (HX1 : equations_well_formed s1).
  { unfold Join_env.join_aliases_into_bindings in Haliases.
    refine (aliases_fold_keeps_well_formed p fuel envs _ _ _ s0 s1 Hwf0
              Haliases HX0).
    intros x index t Ht. rewrite types_of_name_lookup in Ht.
    destruct (Keyed.lookup Index.equal index envs)
      as [[use equations] | ] eqn:Hindex; [ | discriminate ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm' Hfound].
    assert (Huse : Join_env.joined_env s0 index = Some use).
    { apply joined_env_lookup. exists equations. exact Hindex. }
    exact (Hwf0 index use Huse _ _ _ Hfound). }
  assert (Htypes1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types ∧ all_well_formed types).
  { intros var types Hvar.
    unfold Join_env.new_equations_in_joined_envs in Hvar.
    rewrite lookup_filter_bindings in Hvar.
    destruct (lookup var (Join_env.types_in_joined_envs s1))
      as [types1 | ] eqn:Hlookup; [ | discriminate ].
    destruct (negb (is_bound var (Join_env.types_in_joined_envs s0)));
      [ | discriminate ].
    injection Hvar as <-.
    split; [ exact (proj1 (proj2 (proj1 (proj2 Htrue1))) var types1 Hlookup)
           | ].
    intros index t Ht. exact (proj2 HX1 var types1 index t Hlookup Ht). }
  assert (Hholds_types1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types).
  { intros var types Hvar. exact (proj1 (Htypes1 var types Hvar)). }
  assert (Hrelations1 : relations_hold s1 []).
  { intros r relation n [recorded [Hlookup _]]. discriminate. }
  pose proof (later_envs_invariant _ _ Hlater1 Hinvariant0) as Hinvariant1.
  pose proof (later_envs_kinded _ _ Hlater1 Hkinded0) as Hkinded1.
  pose proof (later_envs_well_formed _ _ Hlater1 Hwf0) as Hwf1.
  destruct (delayed_equations_hold source jt choices Hjt fuel s1 s2 _ []
              relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
              Hrelations1 Hdelayed)
    as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (delayed_equations_kinded source jt choices Hjt fuel s1 s2 _ []
                relations2 Hholds_types1 Htrue1 Hinvariant1 Hkinded1
                Hrelations1 Hstate1 Hdelayed) as Hstate2.
  pose proof (delayed_equations_well_formed source jt choices Hjt Hjt_wf
                Hkeeps fuel s1 s2 _ [] relations2 Htypes1 Htrue1 Hinvariant1
                Hkinded1 Hwf1 Hrelations1 Hstate1 HX1 Hdelayed) as HX2.
  pose proof (later_trans _ _ _ Hlater1 Hlater2) as Hlater02.
  assert (Hdefinitions2 :
            ∀ (x : variable)
              (definition : Simples_in_joined_envs.t * flambda_kind),
              lookup x (Join_env.new_definitions_of_existentials s2 s0)
              = Some definition ->
              lookup x (Join_env.definitions_of_existentials s2)
              = Some definition).
  { intros x definition Hx.
    unfold Join_env.new_definitions_of_existentials in Hx.
    rewrite lookup_filter_bindings in Hx.
    destruct (lookup x (Join_env.definitions_of_existentials s2))
      as [definition2 | ]; [ | discriminate ].
    destruct (negb (is_bound x (Join_env.definitions_of_existentials s0)));
      [ | discriminate ].
    exact Hx. }
  exact (loop_well_formed source jt choices Hjt Hjt_wf Hkeeps p Hconst
           Hconst_wf (Join_env.join_depth p) fuel s2 s3 relations2 relations3
           _ Hdefinitions2
           (later_true source s1 s2 Hlater2 Hholds2 (proj2 (proj2 Htrue1)))
           (later_envs_invariant _ _ Hlater02 Hinvariant0)
           (later_envs_kinded _ _ Hlater02 Hkinded0)
           (later_envs_well_formed _ _ Hlater02 Hwf0) Hrelations2 Hstate2 HX2
           Hloop).
Qed.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

Section Define_well_formed.

Context {meet_env meet_expanded_head : Type}
  (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
  (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
  (unique : variable -> bool).

(* The types of the equations and those kept for the variables that
   are removed are well formed. *)
Definition well_formed_invariant (acc : define_acc meet_env) : Prop :=
  let '(_, to_expand, equations, _, _) := acc in
  (∀ (x : variable) (ty : flambda_type),
     lookup x equations = Some ty -> type_well_formed ty = true)
  ∧ ∀ (e : variable) (ty : flambda_type),
      lookup e to_expand = Some ty -> type_well_formed ty = true.

Lemma define_step_well_formed (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  well_formed_invariant acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  well_formed_invariant acc'.
Proof.
  intros Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hequations Hexpand].
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ];
    [ | | discriminate ].
  - destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ] eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    assert (Hequations1 :
              ∀ (x : variable) (ty : flambda_type),
                lookup x equations1 = Some ty -> type_well_formed ty = true).
    { unfold Join_env.move_equation in Hmove.
      destruct (is_bound canon_var equations); [ discriminate | ].
      destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
      - injection Hmove as <-. intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x canon_var).
        + injection Hx as <-. exact (Hequations var ty_var Hlookup_var).
        + rewrite lookup_unbind in Hx.
          destruct (Pos.eqb x var); [ discriminate | ].
          exact (Hequations x ty Hx).
      - injection Hmove as <-. exact Hequations. }
    split.
    + intros x ty Hx.
      destruct (lookup_fold_cons _ _ _ _ _ Hx) as [[_ ->] | Hlookup].
      * apply alias_type_of_well_formed.
      * exact (Hequations1 x ty Hlookup).
    + intros e ty He. rewrite lookup_cons in He.
      destruct (Pos.eqb e var).
      * injection He as <-. apply alias_type_of_well_formed.
      * exact (Hexpand e ty He).
  - destruct (unique var).
    + injection Hstep as <-. split.
      * intros x ty Hx. rewrite lookup_unbind in Hx.
        destruct (Pos.eqb x var); [ discriminate | ].
        exact (Hequations x ty Hx).
      * intros e ty He. rewrite lookup_cons in He.
        destruct (Pos.eqb e var); [ | exact (Hexpand e ty He) ].
        injection He as <-.
        destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
        -- exact (Hequations var ty_var Hlookup_var).
        -- apply unknown_well_formed.
    + injection Hstep as <-. split; [ exact Hequations | exact Hexpand ].
Qed.

Lemma define_fold_well_formed (acc acc' : define_acc meet_env) :
  well_formed_invariant acc ->
  List.fold_left (define_step ops s tenv_of_aliases unique)
    (Bindings_in_target_env.existential_variables_in_order
       (Join_env.bindings s))
    (Some acc)
  = Some acc' ->
  well_formed_invariant acc'.
Proof.
  apply (fold_left_invariant (define_step ops s tenv_of_aliases unique)
           well_formed_invariant (fun b => eq_refl)).
  intros a a' b _ Ha Hstep.
  exact (define_step_well_formed a a' b Ha Hstep).
Qed.

End Define_well_formed.

Theorem last_phase_adds_well_formed : last_phase_adds_well_formed_statement.
Proof.
  intros tenv fuel s relations kept equations relation_equations definitions
    Htypes Hadds n ty Hin.
  destruct (last_phase_adds_parts tenv fuel s relations _ definitions Hadds)
    as [r1 [to_expand [equations1 [relations1 [kept0 [equations0
          [relation_equations0
             [Hfold [_ [_ [_ [_ [Hequations [Hrelation_equations
                                              Heq]]]]]]]]]]]]]].
  injection Heq as _ -> ->.
  assert (Hstart :
            well_formed_invariant
              ({| recorded_variables :=
                    Bindings_in_target_env.imported_variables_in_order
                      (Join_env.bindings s);
                  recorded_extensions := [] |},
               [], Join_env.types_in_target_env s, relations,
               Join_env.definitions_of_existentials s)).
  { split; [ exact Htypes | intros e ty0 He; discriminate ]. }
  destruct (define_fold_well_formed (recording tenv) s tenv
              (unique_of s relations) _ _ Hstart Hfold)
    as [Hequations_inv Hexpand_inv].
  pose proof (expand_variable_well_formed to_expand Hexpand_inv) as Hexp.
  destruct (project_variables_out_well_formed (domain to_expand)
              (Join_env.expand_variable fuel to_expand)
              (fun e ty0 _ He => Hexp fuel e ty0 He))
    as [Htype _].
  apply in_app_or in Hin. destruct Hin as [Hin | Hin].
  - destruct (traverse_in _ _ _ Hequations (n, ty) Hin)
      as [[x ty0] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    destruct (project_variables_out (domain to_expand)
                (Join_env.expand_variable fuel to_expand) ty0) as [ty1 | ]
      eqn:Hty; [ | discriminate ].
    injection Hf as _ <-. apply in_compact in Hbinding.
    exact (Htype ty0 ty1 Hty (Hequations_inv x ty0 Hbinding)).
  - destruct (traverse_in _ _ _ Hrelation_equations (n, ty) Hin)
      as [[r names] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    match type of Hf with
    | context [project_variables_out ?to_project ?expand ?ty0] =>
      destruct (project_variables_out to_project expand ty0) as [ty1 | ]
        eqn:Hty; [ | discriminate ]
    end.
    injection Hf as _ <-.
    exact (Htype _ ty1 Hty eq_refl).
Qed.

(* ======================================================================== *)
(* What the join is given in FlambdaJoinMeet.v                              *)
(* ======================================================================== *)

Theorem n_way_join_of_types_keeps_well_formed
  : n_way_join_of_types_keeps_well_formed_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate fuel.
  exact
    (n_way_join_keeps_well_formed p fuel_of_env fuel expand_head
       reduce_head_of_kind_naked_immediate FlambdaTypeEqualSpec.type_eqb
       FlambdaTypeEqualSpec.block_shape_eqb).
Qed.

Theorem join_parameters_well_formed : join_parameters_well_formed_statement.
Proof.
  intros join_depth c. cbn [Join_env.expand_const join_parameters].
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try reflexivity.
  destruct k as [ | [] | | ]; reflexivity.
Qed.

Theorem expand_head_for_join_well_formed
  : expand_head_for_join_well_formed_statement.
Proof.
  intros tenv ty expanded Hexpand Htypes Hty.
  unfold expand_head_for_join in Hexpand.
  destruct (FlambdaExpandHead.expand_head tenv ty) as [expanded0 | ] eqn:He;
    [ | discriminate ].
  injection Hexpand as <-.
  assert (Hto :
            FlambdaJoinTypes.Expanded_type.to_type (join_expanded expanded0)
            = FlambdaExpandHead.Expanded_type.to_type expanded0).
  { destruct expanded0
      as [k [ | [head | head | k' head | head | head] | ]]; reflexivity. }
  rewrite Hto.
  exact (FlambdaExpandHeadProofs.expand_head_well_formed tenv ty expanded0 He
           Htypes Hty).
Qed.
