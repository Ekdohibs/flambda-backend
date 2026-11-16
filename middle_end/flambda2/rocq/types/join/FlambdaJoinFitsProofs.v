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


(* The proofs of the statements of FlambdaJoinFitsSpec.v. This file
   needs no review.

   The three statements about what a projected type mentions are one
   induction over the seven functions of the projection
   ([project_all_passes]), for a list that is made from a type as
   [free_variables] is: the variables, the aliases with their kinds,
   the names of equations with their kinds.

   The statement about the last phase is by an invariant of its fold
   over the variables that the join has made ([scope_invariant]): the
   equations kept, the types put for the variables removed and the
   relations mention only variables of the target so far, and the
   equations and the relations are on variables that are not
   removed.

   The statements about the first phases are by a relation between
   states ([keeps_scope]) that every function of the join of
   environments keeps: the state grows, and if the types of its
   equations mention only variables of the target but for some that
   are still to be brought in ([scoped_but]), they still do. (The
   join adds an equation before it brings in the variables of its
   type.) The rounds are then as in FlambdaJoinStateProofs.v, whose
   lemmas give the invariant at each step. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeProject FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinDefineSpec
  FlambdaJoinDefineProofs FlambdaJoinTypes FlambdaJoinTypesKeepsSpec
  FlambdaJoinTypesKeepsProofs FlambdaJoinFitsSpec.
Import ListNotations.

(* ---------------------------------------------------------------------- *)
(* A list made from a type as the free variables are                       *)
(* ---------------------------------------------------------------------- *)

Section Project_observed.

Context (to_project : list variable)
  (expand : variable -> option flambda_type).

(* What is observed of a type: a list, made as [free_variables] is. *)
Context {O : Type} (obs : flambda_type -> list O)
  (obs_head : head_of_kind_value -> list O)
  (obs_non_null : head_of_kind_value_non_null -> list O)
  (obs_blocks : row_like_for_blocks -> list O)
  (obs_closures : row_like_for_closures -> list O)
  (obs_entry : closures_entry -> list O)
  (obs_extension : env_extension -> list O)
  (of_alias : flambda_kind -> simple -> list O)
  (of_relation : option variable -> list O)
  (of_naked_immediate : head_of_kind_naked_immediate -> list O)
  (of_equation : name -> flambda_type -> list O).

Definition obs_descr {head : Type} (k : flambda_kind)
    (obs_of_head : head -> list O) (ty : type_descr head) : list O :=
  match ty with
  | Ok (No_alias head) => obs_of_head head
  | Ok (Equals s) => of_alias k s
  | Unknown | Bottom => []
  end.

Definition obs_case (case : row_like_block_case) : list O :=
  flat_map obs (case_maps_to case) ++ obs_extension (case_env_extension case).

Hypothesis obs_value :
  ∀ ty : type_descr head_of_kind_value,
    obs (Type_value ty) = obs_descr Value obs_head ty.
Hypothesis obs_naked_immediate :
  ∀ ty : type_descr head_of_kind_naked_immediate,
    obs (Type_naked_immediate ty)
    = obs_descr (Naked_number Kind_naked_immediate) of_naked_immediate ty.
Hypothesis obs_naked_number :
  ∀ (k : naked_number_kind) (ty : type_descr (list Z)),
    obs (Type_naked_number k ty)
    = obs_descr (Naked_number k) (fun _ => []) ty.
Hypothesis obs_rec_info :
  ∀ ty : type_descr unit,
    obs (Type_rec_info ty) = obs_descr Rec_info (fun _ => []) ty.
Hypothesis obs_region :
  ∀ ty : type_descr unit,
    obs (Type_region ty) = obs_descr Region (fun _ => []) ty.
Hypothesis obs_head_eq :
  ∀ (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null),
    obs_head (Head_of_kind_value non_null is_null)
    = match non_null with
      | Ok head => obs_non_null head
      | Unknown | Bottom => []
      end
      ++ match is_null with
         | Not_null => []
         | Maybe_null is_null => of_relation is_null
         end.
Hypothesis obs_variant :
  ∀ (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool),
    obs_non_null
      (Variant is_int immediates get_tag blocks extensions is_unique)
    = of_relation is_int
      ++ match immediates with
         | Or_unknown.Known ty => obs ty
         | Or_unknown.Unknown => []
         end
      ++ of_relation get_tag
      ++ match blocks with
         | Or_unknown.Known blocks => obs_blocks blocks
         | Or_unknown.Unknown => []
         end
      ++ match extensions with
         | No_extensions => []
         | Ext when_immediate when_block =>
           obs_extension when_immediate ++ obs_extension when_block
         end.
Hypothesis obs_mutable_block :
  ∀ alloc_mode : Alloc_mode.t, obs_non_null (Mutable_block alloc_mode) = [].
Hypothesis obs_string :
  ∀ strs : list (list Z), obs_non_null (String strs) = [].
Hypothesis obs_boxed_number :
  ∀ (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t),
    obs_non_null (Boxed_number k contents alloc_mode) = obs contents.
Hypothesis obs_closures_eq :
  ∀ (by_function_slot : row_like_for_closures) (alloc_mode : Alloc_mode.t),
    obs_non_null (Closures by_function_slot alloc_mode)
    = obs_closures by_function_slot.
Hypothesis obs_blocks_eq :
  ∀ (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t),
    obs_blocks (Row_like_for_blocks known_tags other_tags alloc_mode)
    = flat_map
        (fun case =>
           match snd case with
           | Or_unknown.Known case => obs_case case
           | Or_unknown.Unknown => []
           end)
        known_tags
      ++ match other_tags with
         | Or_bottom.Ok case => obs_case case
         | Or_bottom.Bottom => []
         end.
Hypothesis obs_row_closures :
  ∀ known_closures : finite_map row_like_closures_case,
    obs_closures (Row_like_for_closures known_closures)
    = flat_map
        (fun case =>
           obs_entry (case_maps_to (snd case))
           ++ obs_extension (case_env_extension (snd case)))
        known_closures.
Hypothesis obs_entry_eq :
  ∀ (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type),
    obs_entry (Closures_entry function_types closure_types value_slot_types)
    = flat_map (fun binding => obs (snd binding)) closure_types
      ++ flat_map (fun binding => obs (snd binding)) value_slot_types.
Hypothesis obs_extension_eq :
  ∀ equations : Keyed.finite_map name flambda_type,
    obs_extension (Env_extension equations)
    = flat_map
        (fun binding =>
           of_equation (fst binding) (snd binding) ++ obs (snd binding))
        equations.

(* What is supposed of the things observed before, and what is to be
   shown of those observed after. *)
Context (Before After : O -> Prop).

Definition passes (before after : list O) : Prop :=
  (∀ o : O, In o before -> Before o) -> ∀ o : O, In o after -> After o.

Lemma passes_nil (before : list O) : passes before [].
Proof. intros _ o []. Qed.

Lemma passes_app (before1 before2 after1 after2 : list O) :
  passes before1 after1 ->
  passes before2 after2 ->
  passes (before1 ++ before2) (after1 ++ after2).
Proof.
  intros H1 H2 Hbefore o Ho. apply in_app_or in Ho. destruct Ho as [Ho | Ho].
  - apply H1; [ | exact Ho ]. intros o' Ho'. apply Hbefore. apply in_or_app.
    left. exact Ho'.
  - apply H2; [ | exact Ho ]. intros o' Ho'. apply Hbefore. apply in_or_app.
    right. exact Ho'.
Qed.

Lemma passes_traverse {A B : Type} (f : A -> option B) (g : A -> list O)
    (g' : B -> list O) :
  ∀ (l : list A) (l' : list B),
    traverse f l = Some l' ->
    (∀ (a : A) (b : B), In a l -> f a = Some b -> passes (g a) (g' b)) ->
    passes (flat_map g l) (flat_map g' l').
Proof.
  induction l as [ | a l IH]; intros l' H Hf; cbn [traverse] in H.
  - injection H as <-. apply passes_nil.
  - destruct (f a) as [b | ] eqn:Ha; [ | discriminate ].
    destruct (traverse f l) as [l0 | ] eqn:Hl; [ | discriminate ].
    injection H as <-. cbn [flat_map]. apply passes_app.
    + exact (Hf a b (or_introl eq_refl) Ha).
    + apply IH; [ reflexivity | ]. intros a0 b0 Hin.
      exact (Hf a0 b0 (or_intror Hin)).
Qed.

Lemma passes_traverse_concat {A B : Type} (f : A -> option (list B))
    (g : A -> list O) (g' : B -> list O) :
  ∀ (l : list A) (lists : list (list B)),
    traverse f l = Some lists ->
    (∀ (a : A) (bs : list B),
       In a l -> f a = Some bs -> passes (g a) (flat_map g' bs)) ->
    passes (flat_map g l) (flat_map g' (concat lists)).
Proof.
  induction l as [ | a l IH]; intros lists H Hf; cbn [traverse] in H.
  - injection H as <-. apply passes_nil.
  - destruct (f a) as [bs | ] eqn:Ha; [ | discriminate ].
    destruct (traverse f l) as [lists0 | ] eqn:Hl; [ | discriminate ].
    injection H as <-. cbn [flat_map concat]. rewrite flat_map_app.
    apply passes_app.
    + exact (Hf a bs (or_introl eq_refl) Ha).
    + apply IH; [ reflexivity | ]. intros a0 bs0 Hin.
      exact (Hf a0 bs0 (or_intror Hin)).
Qed.

Hypothesis alias_kept :
  ∀ (k : flambda_kind) (s : simple),
    match s with
    | Var var => among var to_project = false
    | Symbol _ | Const _ => True
    end ->
    passes (of_alias k s) (of_alias k s).
Hypothesis alias_projected :
  ∀ (k : flambda_kind) (var : variable) (expanded : flambda_type),
    among var to_project = true ->
    expand var = Some expanded ->
    kind expanded = k ->
    passes (of_alias k (Var var)) (obs expanded).
Hypothesis relation_passes :
  ∀ x x' : option variable,
    project_relation to_project expand x = Some x' ->
    passes (of_relation x) (of_relation x').
Hypothesis naked_immediate_passes :
  ∀ head head' : head_of_kind_naked_immediate,
    project_head_of_kind_naked_immediate to_project expand head = Some head' ->
    passes (of_naked_immediate head) (of_naked_immediate head').
Hypothesis equation_kept :
  ∀ (n : name) (ty ty' : flambda_type),
    match n with
    | Name_var var => among var to_project = false
    | Name_symbol _ => True
    end ->
    kind ty' = kind ty ->
    passes (of_equation n ty) (of_equation n ty').

Lemma project_descr_passes {head : Type} (k : flambda_kind)
    (project_head : head -> option head)
    (of_expanded : flambda_type -> option (type_descr head))
    (wrap : type_descr head -> flambda_type) (obs_of_head : head -> list O)
    (ty ty' : type_descr head) :
  (∀ d : type_descr head, obs (wrap d) = obs_descr k obs_of_head d) ->
  (∀ (expanded : flambda_type) (d : type_descr head),
     of_expanded expanded = Some d -> expanded = wrap d ∧ kind expanded = k) ->
  (∀ h h' : head,
     ty = Ok (No_alias h) ->
     project_head h = Some h' ->
     passes (obs_of_head h) (obs_of_head h')) ->
  project_descr to_project expand project_head of_expanded ty = Some ty' ->
  passes (obs (wrap ty)) (obs (wrap ty')).
Proof.
  intros Hobs Hof Hhead. unfold project_descr. rewrite (Hobs ty).
  destruct ty as [ | [h | [e | sym | c]] | ]; cbn [obs_descr].
  - intros H. injection H as <-. rewrite Hobs. apply passes_nil.
  - destruct (project_head h) as [h' | ] eqn:Hh; [ | discriminate ].
    intros H. injection H as <-. rewrite Hobs. cbn [obs_descr].
    exact (Hhead h h' eq_refl Hh).
  - destruct (among e to_project) eqn:Hproject.
    + destruct (expand e) as [expanded | ] eqn:Hexpand; [ | discriminate ].
      intros Hd. destruct (Hof _ _ Hd) as [Hwrap Hkind]. rewrite <- Hwrap.
      exact (alias_projected k e expanded Hproject Hexpand Hkind).
    + intros H. injection H as <-. rewrite Hobs. cbn [obs_descr].
      exact (alias_kept k (Var e) Hproject).
  - intros H. injection H as <-. rewrite Hobs. cbn [obs_descr].
    exact (alias_kept k (Symbol sym) I).
  - intros H. injection H as <-. rewrite Hobs. cbn [obs_descr].
    exact (alias_kept k (Const c) I).
  - intros H. injection H as <-. rewrite Hobs. apply passes_nil.
Qed.


Lemma project_kind (t t' : flambda_type) :
  project_variables_out to_project expand t = Some t' -> kind t' = kind t.
Proof.
  destruct t as [d | d | k d | d | d]; simpl;
    match goal with
    | |- context [project_descr ?a ?b ?c ?e ?f] =>
      destruct (project_descr a b c e f); [ | discriminate ]
    end;
    intros H; injection H as <-; reflexivity.
Qed.

Lemma project_all_passes :
  (∀ t t' : flambda_type,
     project_variables_out to_project expand t = Some t' ->
     passes (obs t) (obs t'))
  ∧ (∀ head head' : head_of_kind_value,
       project_head_of_kind_value to_project expand head = Some head' ->
       passes (obs_head head) (obs_head head'))
  ∧ (∀ head head' : head_of_kind_value_non_null,
       project_head_of_kind_value_non_null to_project expand head
       = Some head' ->
       passes (obs_non_null head) (obs_non_null head'))
  ∧ (∀ blocks blocks' : row_like_for_blocks,
       project_row_like_for_blocks to_project expand blocks = Some blocks' ->
       passes (obs_blocks blocks) (obs_blocks blocks'))
  ∧ (∀ closures closures' : row_like_for_closures,
       project_row_like_for_closures to_project expand closures
       = Some closures' ->
       passes (obs_closures closures) (obs_closures closures'))
  ∧ (∀ entry entry' : closures_entry,
       project_closures_entry to_project expand entry = Some entry' ->
       passes (obs_entry entry) (obs_entry entry'))
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       passes (obs_extension extension) (obs_extension extension')).
Proof.
  apply (types_induction
           (fun t =>
              ∀ t' : flambda_type,
                project_variables_out to_project expand t = Some t' ->
                passes (obs t) (obs t'))
           (fun head =>
              ∀ head' : head_of_kind_value,
                project_head_of_kind_value to_project expand head
                = Some head' ->
                passes (obs_head head) (obs_head head'))
           (fun head =>
              ∀ head' : head_of_kind_value_non_null,
                project_head_of_kind_value_non_null to_project expand head
                = Some head' ->
                passes (obs_non_null head) (obs_non_null head'))
           (fun blocks =>
              ∀ blocks' : row_like_for_blocks,
                project_row_like_for_blocks to_project expand blocks
                = Some blocks' ->
                passes (obs_blocks blocks) (obs_blocks blocks'))
           (fun closures =>
              ∀ closures' : row_like_for_closures,
                project_row_like_for_closures to_project expand closures
                = Some closures' ->
                passes (obs_closures closures) (obs_closures closures'))
           (fun entry =>
              ∀ entry' : closures_entry,
                project_closures_entry to_project expand entry = Some entry'
                ->
                passes (obs_entry entry) (obs_entry entry'))
           (fun extension =>
              ∀ extension' : env_extension,
                project_env_extension to_project expand extension
                = Some extension' ->
                passes (obs_extension extension) (obs_extension extension'))).
  - (* Type_value *)
    intros ty Hhead t' H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_value to_project expand)
                value_of_expanded ty) as [ty' | ] eqn:Hd; [ | discriminate ].
    injection H as <-.
    refine (project_descr_passes Value _ _ Type_value obs_head ty ty'
              obs_value _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. split; reflexivity.
    + intros h h' Heq Hh. subst ty. exact (Hhead h' Hh).
  - (* Type_naked_immediate *)
    intros ty t' H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_naked_immediate to_project expand)
                naked_immediate_of_expanded ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-.
    refine (project_descr_passes (Naked_number Kind_naked_immediate) _ _
              Type_naked_immediate of_naked_immediate ty ty'
              obs_naked_immediate _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. split; reflexivity.
    + intros h h' _ Hh. exact (naked_immediate_passes h h' Hh).
  - (* Type_naked_number *)
    intros k ty t' H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                (naked_number_of_expanded k) ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-.
    refine (project_descr_passes (Naked_number k) _ _ (Type_naked_number k)
              (fun _ => []) ty ty' (obs_naked_number k) _ _ Hd).
    + intros expanded d Hof.
      rewrite (of_expanded_naked_number k expanded d Hof).
      split; reflexivity.
    + intros h h' _ Hh. apply passes_nil.
  - (* Type_rec_info *)
    intros ty t' H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                rec_info_of_expanded ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-.
    refine (project_descr_passes Rec_info _ _ Type_rec_info (fun _ => []) ty
              ty' obs_rec_info _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. split; reflexivity.
    + intros h h' _ Hh. apply passes_nil.
  - (* Type_region *)
    intros ty t' H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                region_of_expanded ty) as [ty' | ] eqn:Hd; [ | discriminate ].
    injection H as <-.
    refine (project_descr_passes Region _ _ Type_region (fun _ => []) ty ty'
              obs_region _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. split; reflexivity.
    + intros h h' _ Hh. apply passes_nil.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null head' H. simpl in H.
    assert (Hnull :
              ∀ is_null' : FlambdaTypeGrammar.is_null,
                match is_null with
                | Not_null => Some Not_null
                | Maybe_null is_null_var =>
                  let* is_null_var :=
                    project_relation to_project expand is_null_var
                  in
                  Some (Maybe_null is_null_var)
                end
                = Some is_null' ->
                passes
                  match is_null with
                  | Not_null => []
                  | Maybe_null is_null => of_relation is_null
                  end
                  match is_null' with
                  | Not_null => []
                  | Maybe_null is_null => of_relation is_null
                  end).
    { intros is_null' Hn. destruct is_null as [ | x].
      - injection Hn as <-. apply passes_nil.
      - destruct (project_relation to_project expand x) as [x' | ] eqn:Hx;
          [ | discriminate ].
        injection Hn as <-. exact (relation_passes x x' Hx). }
    destruct (match is_null with
              | Not_null => Some Not_null
              | Maybe_null is_null_var =>
                let* is_null_var :=
                  project_relation to_project expand is_null_var
                in
                Some (Maybe_null is_null_var)
              end) as [is_null' | ] eqn:His_null; [ | discriminate ].
    assert (Hhead :
              ∀ non_null' : or_unknown_or_bottom head_of_kind_value_non_null,
                match non_null with
                | Unknown => Some Unknown
                | Bottom => Some Bottom
                | Ok non_null_head =>
                  let* non_null_head :=
                    project_head_of_kind_value_non_null to_project expand
                      non_null_head
                  in
                  Some (Ok non_null_head)
                end
                = Some non_null' ->
                passes
                  match non_null with
                  | Ok head => obs_non_null head
                  | Unknown | Bottom => []
                  end
                  match non_null' with
                  | Ok head => obs_non_null head
                  | Unknown | Bottom => []
                  end).
    { intros non_null' Hn. destruct non_null as [ | head | ].
      - injection Hn as <-. apply passes_nil.
      - destruct (project_head_of_kind_value_non_null to_project expand head)
          as [head0 | ] eqn:Hhead0; [ | discriminate ].
        injection Hn as <-. exact (Hnon_null head0 eq_refl).
      - injection Hn as <-. apply passes_nil. }
    destruct (match non_null with
              | Unknown => Some Unknown
              | Bottom => Some Bottom
              | Ok non_null_head =>
                let* non_null_head :=
                  project_head_of_kind_value_non_null to_project expand
                    non_null_head
                in
                Some (Ok non_null_head)
              end) as [non_null' | ] eqn:Hnon_null'; [ | discriminate ].
    injection H as <-. rewrite !obs_head_eq.
    exact (passes_app _ _ _ _ (Hhead non_null' eq_refl)
             (Hnull is_null' eq_refl)).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions head' H.
    simpl in H.
    destruct (project_relation to_project expand is_int) as [is_int' | ]
      eqn:His_int; [ | discriminate ].
    destruct (project_relation to_project expand get_tag) as [get_tag' | ]
      eqn:Hget_tag; [ | discriminate ].
    assert (Himm :
              ∀ immediates' : Or_unknown.t flambda_type,
                match immediates with
                | Or_unknown.Known immediates =>
                  let* immediates :=
                    project_variables_out to_project expand immediates
                  in
                  Some (Or_unknown.Known immediates)
                | Or_unknown.Unknown => Some Or_unknown.Unknown
                end
                = Some immediates' ->
                passes
                  match immediates with
                  | Or_unknown.Known ty => obs ty
                  | Or_unknown.Unknown => []
                  end
                  match immediates' with
                  | Or_unknown.Known ty => obs ty
                  | Or_unknown.Unknown => []
                  end).
    { intros immediates' Hi. destruct immediates as [imm | ].
      - destruct (project_variables_out to_project expand imm) as [imm' | ]
          eqn:Himm'; [ | discriminate ].
        injection Hi as <-. exact (Himmediates imm' eq_refl).
      - injection Hi as <-. apply passes_nil. }
    destruct (match immediates with
              | Or_unknown.Known immediates =>
                let* immediates :=
                  project_variables_out to_project expand immediates
                in
                Some (Or_unknown.Known immediates)
              | Or_unknown.Unknown => Some Or_unknown.Unknown
              end) as [immediates' | ] eqn:Himmediates'; [ | discriminate ].
    assert (Hrow :
              ∀ blocks' : Or_unknown.t row_like_for_blocks,
                match blocks with
                | Or_unknown.Known blocks =>
                  let* blocks :=
                    project_row_like_for_blocks to_project expand blocks
                  in
                  Some (Or_unknown.Known blocks)
                | Or_unknown.Unknown => Some Or_unknown.Unknown
                end
                = Some blocks' ->
                passes
                  match blocks with
                  | Or_unknown.Known blocks => obs_blocks blocks
                  | Or_unknown.Unknown => []
                  end
                  match blocks' with
                  | Or_unknown.Known blocks => obs_blocks blocks
                  | Or_unknown.Unknown => []
                  end).
    { intros blocks' Hb. destruct blocks as [row | ].
      - destruct (project_row_like_for_blocks to_project expand row)
          as [row' | ] eqn:Hrow'; [ | discriminate ].
        injection Hb as <-. exact (Hblocks row' eq_refl).
      - injection Hb as <-. apply passes_nil. }
    destruct (match blocks with
              | Or_unknown.Known blocks =>
                let* blocks :=
                  project_row_like_for_blocks to_project expand blocks
                in
                Some (Or_unknown.Known blocks)
              | Or_unknown.Unknown => Some Or_unknown.Unknown
              end) as [blocks' | ] eqn:Hblocks'; [ | discriminate ].
    assert (Hext :
              ∀ extensions' : variant_extensions,
                match extensions with
                | No_extensions => Some No_extensions
                | Ext when_immediate when_block =>
                  let* when_immediate :=
                    project_env_extension to_project expand when_immediate
                  in
                  let* when_block :=
                    project_env_extension to_project expand when_block
                  in
                  Some (Ext when_immediate when_block)
                end
                = Some extensions' ->
                passes
                  match extensions with
                  | No_extensions => []
                  | Ext when_immediate when_block =>
                    obs_extension when_immediate ++ obs_extension when_block
                  end
                  match extensions' with
                  | No_extensions => []
                  | Ext when_immediate when_block =>
                    obs_extension when_immediate ++ obs_extension when_block
                  end).
    { intros extensions' He. destruct extensions as [ | wi wb].
      - injection He as <-. apply passes_nil.
      - destruct (project_env_extension to_project expand wi) as [wi' | ]
          eqn:Hwi; [ | discriminate ].
        destruct (project_env_extension to_project expand wb) as [wb' | ]
          eqn:Hwb; [ | discriminate ].
        injection He as <-. destruct Hextensions as [IHwi IHwb].
        exact (passes_app _ _ _ _ (IHwi wi' eq_refl) (IHwb wb' eq_refl)). }
    destruct (match extensions with
              | No_extensions => Some No_extensions
              | Ext when_immediate when_block =>
                let* when_immediate :=
                  project_env_extension to_project expand when_immediate
                in
                let* when_block :=
                  project_env_extension to_project expand when_block
                in
                Some (Ext when_immediate when_block)
              end) as [extensions' | ] eqn:Hextensions'; [ | discriminate ].
    injection H as <-. rewrite !obs_variant.
    apply passes_app; [ exact (relation_passes _ _ His_int) | ].
    apply passes_app; [ exact (Himm immediates' eq_refl) | ].
    apply passes_app; [ exact (relation_passes _ _ Hget_tag) | ].
    apply passes_app; [ exact (Hrow blocks' eq_refl) | ].
    exact (Hext extensions' eq_refl).
  - (* Mutable_block *)
    intros alloc_mode head' H. injection H as <-. rewrite obs_mutable_block.
    apply passes_nil.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents head' H. simpl in H.
    destruct (project_variables_out to_project expand contents)
      as [contents' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. rewrite !obs_boxed_number.
    exact (Hcontents contents' eq_refl).
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures head' H. simpl in H.
    destruct (project_row_like_for_closures to_project expand
                by_function_slot) as [closures' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. rewrite !obs_closures_eq.
    exact (Hclosures closures' eq_refl).
  - (* String *)
    intros strs head' H. injection H as <-. rewrite obs_string.
    apply passes_nil.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother blocks' H.
    simpl in H.
    assert (Hcase :
              ∀ case case' : row_like_block_case,
                P_block_case
                  (fun t =>
                     ∀ t' : flambda_type,
                       project_variables_out to_project expand t = Some t' ->
                       passes (obs t) (obs t'))
                  (fun extension =>
                     ∀ extension' : env_extension,
                       project_env_extension to_project expand extension
                       = Some extension' ->
                       passes (obs_extension extension)
                         (obs_extension extension'))
                  case ->
                project_block_case (project_variables_out to_project expand)
                  (project_env_extension to_project expand) case
                = Some case' ->
                passes (obs_case case) (obs_case case')).
    { intros case case' [Htypes Hextension] Hp.
      unfold project_block_case in Hp.
      destruct (project_env_extension to_project expand
                  (case_env_extension case)) as [ext' | ] eqn:Hext';
        [ | discriminate ].
      destruct (traverse (project_variables_out to_project expand)
                  (case_maps_to case)) as [maps' | ] eqn:Hmaps;
        [ | discriminate ].
      injection Hp as <-. unfold obs_case.
      cbn [case_maps_to case_env_extension]. apply passes_app.
      - apply (passes_traverse _ _ _ _ _ Hmaps).
        intros ty ty' Hin Hty. exact (Htypes ty Hin ty' Hty).
      - exact (Hextension ext' eq_refl). }
    match type of H with
    | context [traverse ?f known_tags] =>
      destruct (traverse f known_tags) as [known_tags' | ] eqn:Htraverse;
        [ | discriminate ]
    end.
    assert (Hother' :
              ∃ other_tags' : Or_bottom.t row_like_block_case,
                Some (Row_like_for_blocks known_tags' other_tags' alloc_mode)
                = Some blocks'
                ∧ passes
                    match other_tags with
                    | Or_bottom.Ok case => obs_case case
                    | Or_bottom.Bottom => []
                    end
                    match other_tags' with
                    | Or_bottom.Ok case => obs_case case
                    | Or_bottom.Bottom => []
                    end).
    { destruct other_tags as [case | ].
      - destruct (project_block_case _ _ case) as [case' | ] eqn:Hcase';
          [ | discriminate ].
        exists (Or_bottom.Ok case'). split; [ exact H | ].
        exact (Hcase case case' Hother Hcase').
      - exists Or_bottom.Bottom. split; [ exact H | apply passes_nil ]. }
    clear H. destruct Hother' as [other_tags' [H Hother']].
    injection H as <-. rewrite !obs_blocks_eq. apply passes_app;
      [ | exact Hother' ].
    apply (passes_traverse _ _ _ _ _ Htraverse).
    intros [tag [case | ]] b' Hin Hb; cbn [fst snd] in Hb.
    + destruct (project_block_case _ _ case) as [case' | ] eqn:Hcase';
        [ | discriminate ].
      injection Hb as <-. cbn [snd].
      exact (Hcase case case' (Hknown tag case Hin) Hcase').
    + injection Hb as <-. apply passes_nil.
  - (* Row_like_for_closures *)
    intros known_closures Hknown closures' H. simpl in H.
    match type of H with
    | context [traverse ?f known_closures] =>
      destruct (traverse f known_closures) as [known' | ] eqn:Htraverse;
        [ | discriminate ]
    end.
    injection H as <-. rewrite !obs_row_closures.
    apply (passes_traverse _ _ _ _ _ Htraverse).
    intros [slot case] b' Hin Hb; cbn [fst snd] in Hb.
    destruct (project_env_extension to_project expand
                (case_env_extension case)) as [ext' | ] eqn:Hext';
      [ | discriminate ].
    destruct (project_closures_entry to_project expand (case_maps_to case))
      as [maps' | ] eqn:Hmaps; [ | discriminate ].
    injection Hb as <-. cbn [snd case_maps_to case_env_extension].
    destruct (Hknown slot case Hin) as [IHentry IHext].
    exact (passes_app _ _ _ _ (IHentry maps' Hmaps) (IHext ext' Hext')).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      entry' H.
    simpl in H.
    match type of H with
    | context [traverse ?f closure_types] =>
      destruct (traverse f closure_types) as [closure_types' | ]
        eqn:Htraverse_c; [ | discriminate ]
    end.
    match type of H with
    | context [traverse ?f value_slot_types] =>
      destruct (traverse f value_slot_types) as [value_slot_types' | ]
        eqn:Htraverse_s; [ | discriminate ]
    end.
    injection H as <-. rewrite !obs_entry_eq. apply passes_app.
    + apply (passes_traverse _ _ _ _ _ Htraverse_c).
      intros [g ty] b' Hin Hb; cbn [fst snd] in Hb.
      destruct (project_variables_out to_project expand ty) as [ty' | ]
        eqn:Hty; [ | discriminate ].
      injection Hb as <-. cbn [snd]. exact (Hclosures g ty Hin ty' Hty).
    + apply (passes_traverse _ _ _ _ _ Htraverse_s).
      intros [w ty] b' Hin Hb; cbn [fst snd] in Hb.
      destruct (project_variables_out to_project expand ty) as [ty' | ]
        eqn:Hty; [ | discriminate ].
      injection Hb as <-. cbn [snd]. exact (Hslots w ty Hin ty' Hty).
  - (* Env_extension *)
    intros equations Htypes extension' H. simpl in H.
    match type of H with
    | context [traverse ?f equations] =>
      destruct (traverse f equations) as [lists | ] eqn:Htraverse;
        [ | discriminate ]
    end.
    injection H as <-. rewrite !obs_extension_eq.
    apply (passes_traverse_concat _ _ _ _ _ Htraverse).
    intros [n ty] bs Hin Hb; cbn [fst snd] in Hb.
    assert (Hkeep :
              (let* ty0 := project_variables_out to_project expand ty in
               Some [(n, ty0)])
              = Some bs ->
              match n with
              | Name_var var => among var to_project = false
              | Name_symbol _ => True
              end ->
              passes (of_equation n ty ++ obs ty)
                (flat_map
                   (fun binding =>
                      of_equation (fst binding) (snd binding)
                      ++ obs (snd binding))
                   bs)).
    { intros Hk Hn.
      destruct (project_variables_out to_project expand ty) as [ty' | ]
        eqn:Hty; [ | discriminate ].
      injection Hk as <-. cbn [flat_map fst snd]. rewrite app_nil_r.
      apply passes_app.
      - exact (equation_kept n ty ty' Hn (project_kind ty ty' Hty)).
      - exact (Htypes n ty Hin ty' Hty). }
    destruct n as [var | sym]; cbn [fst snd].
    + destruct (among var to_project) eqn:Hproject.
      * injection Hb as <-. apply passes_nil.
      * exact (Hkeep Hb eq_refl).
    + exact (Hkeep Hb I).
Qed.

End Project_observed.

(* ---------------------------------------------------------------------- *)
(* The variables of a projected type                                       *)
(* ---------------------------------------------------------------------- *)

Section Project_scope.

Context (to_project : list variable)
  (expand : variable -> option flambda_type) (in_target : variable -> Prop).

Hypothesis expand_in_target :
  ∀ (e : variable) (ty : flambda_type) (x : variable),
    among e to_project = true ->
    expand e = Some ty ->
    In x (free_variables ty) ->
    in_target x.

Definition kept_in_target (x : variable) : Prop :=
  among x to_project = false -> in_target x.

Lemma project_all_scope :
  (∀ t t' : flambda_type,
     project_variables_out to_project expand t = Some t' ->
     (∀ x : variable, In x (free_variables t) -> kept_in_target x) ->
     ∀ x : variable, In x (free_variables t') -> in_target x)
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       (∀ x : variable,
          In x (free_variables_of_env_extension extension) ->
          kept_in_target x) ->
       ∀ x : variable,
         In x (free_variables_of_env_extension extension') -> in_target x).
Proof.
  assert (Hall :=
    project_all_passes to_project expand free_variables
      free_variables_of_head_of_kind_value
      free_variables_of_head_of_kind_value_non_null
      free_variables_of_row_like_for_blocks
      free_variables_of_row_like_for_closures
      free_variables_of_closures_entry free_variables_of_env_extension
      (fun _ s => free_variables_of_simple s) free_variables_of_option
      free_variables_of_naked_immediate
      (fun n _ => free_variables_of_name n)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun k ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun non_null is_null => eq_refl)).
  specialize
    (Hall (fun is_int immediates get_tag blocks extensions u => eq_refl)).
  specialize (Hall (fun m => eq_refl)).
  specialize (Hall (fun s => eq_refl)).
  specialize (Hall (fun k c m => eq_refl)).
  specialize (Hall (fun c m => eq_refl)).
  specialize (Hall (fun known other m => eq_refl)).
  specialize (Hall (fun known => eq_refl)).
  specialize (Hall (fun f c v => eq_refl)).
  specialize (Hall (fun equations => eq_refl)).
  specialize (Hall kept_in_target in_target).
  assert (Hone : ∀ x : variable,
            among x to_project = false ->
            passes kept_in_target in_target [x] [x]).
  { intros x Hx Hbefore o [<- | []].
    exact (Hbefore x (or_introl eq_refl) Hx). }
  assert (Hnone : ∀ before : list variable,
            passes kept_in_target in_target before []).
  { intros before _ o []. }
  destruct Hall as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  - (* alias_kept *)
    intros k [x | sym | c] Hs; cbn [free_variables_of_simple].
    + exact (Hone x Hs).
    + apply Hnone.
    + apply Hnone.
  - (* alias_projected *)
    intros k var expanded Hproject Hexpand _ _ o Ho.
    exact (expand_in_target var expanded o Hproject Hexpand Ho).
  - (* relation *)
    intros [e | ] x'; unfold project_relation.
    2:{ intros H. injection H as <-. apply Hnone. }
    destruct (among e to_project) eqn:Hproject; cbn [negb].
    2:{ intros H. injection H as <-. exact (Hone e Hproject). }
    destruct (expand e) as [ty | ] eqn:Hexpand; [ | discriminate ].
    intros H. injection H as <-.
    destruct (get_alias_opt ty) as [[y | sym | c] | ] eqn:Halias;
      try apply Hnone.
    intros _ o [<- | []].
    apply (expand_in_target e ty y Hproject Hexpand).
    exact (get_alias_free_variables ty (Var y) y Halias (or_introl eq_refl)).
  - (* naked immediate *)
    intros head head'. unfold project_head_of_kind_naked_immediate.
    destruct (traverse (project_inverse_relation to_project expand)
                (inverse_relations head)) as [relations | ] eqn:Htraverse;
      [ | discriminate ].
    intros H. injection H as <-. unfold free_variables_of_naked_immediate.
    cbn [inverse_relations].
    apply (passes_traverse_concat kept_in_target in_target _ _ _ _ _
             Htraverse).
    intros [rel n] l _. unfold project_inverse_relation. cbn [fst snd].
    destruct n as [e | sym].
    2:{ intros H. injection H as <-. apply Hnone. }
    destruct (among e to_project) eqn:Hproject; cbn [negb].
    2:{ intros H. injection H as <-. cbn [flat_map snd]. rewrite app_nil_r.
        exact (Hone e Hproject). }
    destruct (expand e) as [ty | ] eqn:Hexpand; [ | discriminate ].
    intros H. injection H as <-.
    destruct (get_alias_opt ty) as [[y | sym | c] | ] eqn:Halias;
      try apply Hnone.
    intros _ o Ho. cbn in Ho. destruct Ho as [<- | []].
    apply (expand_in_target e ty y Hproject Hexpand).
    exact (get_alias_free_variables ty (Var y) y Halias (or_introl eq_refl)).
  - (* equation kept *)
    intros [x | sym] ty ty' Hn _; cbn.
    + exact (Hone x Hn).
    + apply Hnone.
  - split.
    + intros t t' Hproject Hbefore. exact (Htype t t' Hproject Hbefore).
    + intros extension extension' Hproject Hbefore.
      exact (Hextension extension extension' Hproject Hbefore).
Qed.

End Project_scope.

(* ---------------------------------------------------------------------- *)
(* The aliases of a projected type                                         *)
(* ---------------------------------------------------------------------- *)

Section Project_kinded.

Context (to_project : list variable)
  (expand : variable -> option flambda_type)
  (of_kind : simple -> flambda_kind -> Prop).

Hypothesis expand_kinded :
  ∀ (e : variable) (ty : flambda_type) (s : simple) (k : flambda_kind),
    among e to_project = true ->
    expand e = Some ty ->
    In (s, k) (aliases_with_kinds ty) ->
    of_kind s k.

Let alias_before : simple * flambda_kind -> Prop :=
  FlambdaJoinFitsSpec.alias_before to_project expand of_kind.

Definition alias_after (alias : simple * flambda_kind) : Prop :=
  of_kind (fst alias) (snd alias).

Lemma project_all_kinded :
  (∀ t t' : flambda_type,
     project_variables_out to_project expand t = Some t' ->
     (∀ alias : simple * flambda_kind,
        In alias (aliases_with_kinds t) -> alias_before alias) ->
     ∀ (s : simple) (k : flambda_kind),
       In (s, k) (aliases_with_kinds t') -> of_kind s k)
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       (∀ alias : simple * flambda_kind,
          In alias (aliases_of_env_extension extension) -> alias_before alias)
       ->
       ∀ (s : simple) (k : flambda_kind),
         In (s, k) (aliases_of_env_extension extension') -> of_kind s k).
Proof.
  assert (Hall :=
    project_all_passes to_project expand aliases_with_kinds
      aliases_of_head_of_kind_value aliases_of_head_of_kind_value_non_null
      aliases_of_row_like_for_blocks aliases_of_row_like_for_closures
      aliases_of_closures_entry aliases_of_env_extension
      (fun k s => [(s, k)]) aliases_of_relation (fun _ => [])
      (fun _ _ => [])).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun k ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun ty => eq_refl)).
  specialize (Hall (fun non_null is_null => eq_refl)).
  specialize
    (Hall (fun is_int immediates get_tag blocks extensions u => eq_refl)).
  specialize (Hall (fun m => eq_refl)).
  specialize (Hall (fun s => eq_refl)).
  specialize (Hall (fun k c m => eq_refl)).
  specialize (Hall (fun c m => eq_refl)).
  specialize (Hall (fun known other m => eq_refl)).
  specialize (Hall (fun known => eq_refl)).
  specialize (Hall (fun f c v => eq_refl)).
  specialize (Hall (fun equations => eq_refl)).
  specialize (Hall alias_before alias_after).
  assert (Hnone : ∀ before : list (simple * flambda_kind),
            passes alias_before alias_after before []).
  { intros before _ o []. }
  destruct Hall as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  - (* alias_kept *)
    intros k s Hs Hbefore o [<- | []].
    pose proof (Hbefore (s, k) (or_introl eq_refl)) as Hb.
    unfold alias_before, FlambdaJoinFitsSpec.alias_before in Hb.
    unfold alias_after. cbn [fst snd] in *.
    destruct s as [x | sym | c]; [ rewrite Hs in Hb | | ]; exact Hb.
  - (* alias_projected *)
    intros k var expanded Hproject Hexpand _ _ [s k'] Ho.
    exact (expand_kinded var expanded s k' Hproject Hexpand Ho).
  - (* relation *)
    intros [e | ] x'; unfold project_relation.
    2:{ intros H. injection H as <-. apply Hnone. }
    destruct (among e to_project) eqn:Hproject; cbn [negb].
    2:{ intros H. injection H as <-. intros Hbefore o [<- | []].
        pose proof (Hbefore _ (or_introl eq_refl)) as Hb.
        unfold alias_before, FlambdaJoinFitsSpec.alias_before in Hb.
        cbn [fst snd] in Hb.
        rewrite Hproject in Hb. exact Hb. }
    destruct (expand e) as [ty | ] eqn:Hexpand; [ | discriminate ].
    intros H. injection H as <-.
    destruct (get_alias_opt ty) as [[y | sym | c] | ] eqn:Halias;
      try apply Hnone.
    intros Hbefore o [<- | []].
    pose proof (Hbefore _ (or_introl eq_refl)) as Hb.
    unfold alias_before, FlambdaJoinFitsSpec.alias_before in Hb.
    cbn [fst snd] in Hb. rewrite Hproject in Hb.
    unfold alias_after. cbn [fst snd]. rewrite <- (Hb ty Hexpand).
    exact (expand_kinded e ty (Var y) (kind ty) Hproject Hexpand
             (get_alias_in_aliases ty (Var y) Halias)).
  - (* naked immediate *)
    intros head head' _. apply Hnone.
  - (* equation kept *)
    intros n ty ty' _ _. apply Hnone.
  - split.
    + intros t t' Hproject Hbefore s k Hin.
      exact (Htype t t' Hproject Hbefore (s, k) Hin).
    + intros extension extension' Hproject Hbefore s k Hin.
      exact (Hextension extension extension' Hproject Hbefore (s, k) Hin).
Qed.

End Project_kinded.

(* ---------------------------------------------------------------------- *)
(* The names of the equations inside a projected type                      *)
(* ---------------------------------------------------------------------- *)

Section Project_equations_kinded.

Context (to_project : list variable)
  (expand : variable -> option flambda_type)
  (of_kind : name -> flambda_kind -> Prop).

Hypothesis expand_equations_kinded :
  ∀ (e : variable) (ty : flambda_type) (n : name) (k : flambda_kind),
    among e to_project = true ->
    expand e = Some ty ->
    In (n, k) (equation_names_with_kinds ty) ->
    of_kind n k.

Definition equation_before (equation : name * flambda_kind) : Prop :=
  match fst equation with
  | Name_var e =>
    among e to_project = false -> of_kind (fst equation) (snd equation)
  | Name_symbol _ => of_kind (fst equation) (snd equation)
  end.

Definition equation_after (equation : name * flambda_kind) : Prop :=
  of_kind (fst equation) (snd equation).

Lemma project_all_equations_kinded :
  (∀ t t' : flambda_type,
     project_variables_out to_project expand t = Some t' ->
     (∀ equation : name * flambda_kind,
        In equation (equation_names_with_kinds t) -> equation_before equation)
     ->
     ∀ (n : name) (k : flambda_kind),
       In (n, k) (equation_names_with_kinds t') -> of_kind n k)
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       (∀ equation : name * flambda_kind,
          In equation (equation_names_of_env_extension extension) ->
          equation_before equation) ->
       ∀ (n : name) (k : flambda_kind),
         In (n, k) (equation_names_of_env_extension extension') ->
         of_kind n k).
Proof.
  assert (Hall :=
    project_all_passes to_project expand equation_names_with_kinds
      equation_names_of_head_of_kind_value
      equation_names_of_head_of_kind_value_non_null
      equation_names_of_row_like_for_blocks
      equation_names_of_row_like_for_closures
      equation_names_of_closures_entry equation_names_of_env_extension
      (fun _ _ => []) (fun _ => []) (fun _ => [])
      (fun n ty => [(n, kind ty)])).
  assert (H1 : ∀ ty : type_descr head_of_kind_value,
            equation_names_with_kinds (Type_value ty)
            = obs_descr (fun _ _ => []) Value
                equation_names_of_head_of_kind_value ty).
  { intros [ | [h | s] | ]; reflexivity. }
  specialize (Hall H1). clear H1.
  assert (H2 : ∀ ty : type_descr head_of_kind_naked_immediate,
            equation_names_with_kinds (Type_naked_immediate ty)
            = obs_descr (fun _ _ => []) (Naked_number Kind_naked_immediate)
                (fun _ => []) ty).
  { intros [ | [h | s] | ]; reflexivity. }
  specialize (Hall H2). clear H2.
  assert (H3 : ∀ (k : naked_number_kind) (ty : type_descr (list Z)),
            equation_names_with_kinds (Type_naked_number k ty)
            = obs_descr (fun _ _ => []) (Naked_number k) (fun _ => []) ty).
  { intros k [ | [h | s] | ]; reflexivity. }
  specialize (Hall H3). clear H3.
  assert (H4 : ∀ ty : type_descr unit,
            equation_names_with_kinds (Type_rec_info ty)
            = obs_descr (fun _ _ => []) Rec_info (fun _ => []) ty).
  { intros [ | [h | s] | ]; reflexivity. }
  specialize (Hall H4). clear H4.
  assert (H5 : ∀ ty : type_descr unit,
            equation_names_with_kinds (Type_region ty)
            = obs_descr (fun _ _ => []) Region (fun _ => []) ty).
  { intros [ | [h | s] | ]; reflexivity. }
  specialize (Hall H5). clear H5.
  assert (H6 : ∀ (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
              (is_null : is_null),
            equation_names_of_head_of_kind_value
              (Head_of_kind_value non_null is_null)
            = match non_null with
              | Ok head => equation_names_of_head_of_kind_value_non_null head
              | Unknown | Bottom => []
              end
              ++ match is_null with
                 | Not_null => []
                 | Maybe_null _ => []
                 end).
  { intros [ | head | ] [ | x]; cbn; rewrite ?app_nil_r; reflexivity. }
  specialize (Hall H6). clear H6.
  specialize
    (Hall (fun is_int immediates get_tag blocks extensions u => eq_refl)).
  specialize (Hall (fun m => eq_refl)).
  specialize (Hall (fun s => eq_refl)).
  specialize (Hall (fun k c m => eq_refl)).
  specialize (Hall (fun c m => eq_refl)).
  specialize (Hall (fun known other m => eq_refl)).
  specialize (Hall (fun known => eq_refl)).
  specialize (Hall (fun f c v => eq_refl)).
  specialize (Hall (fun equations => eq_refl)).
  specialize (Hall equation_before equation_after).
  assert (Hnone : ∀ before : list (name * flambda_kind),
            passes equation_before equation_after before []).
  { intros before _ o []. }
  destruct Hall as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  - (* alias_kept *)
    intros k s _. apply Hnone.
  - (* alias_projected *)
    intros k var expanded Hproject Hexpand _ _ [n k'] Ho.
    exact (expand_equations_kinded var expanded n k' Hproject Hexpand Ho).
  - (* relation *)
    intros x x' _. apply Hnone.
  - (* naked immediate *)
    intros head head' _. apply Hnone.
  - (* equation kept *)
    intros n ty ty' Hn Hkind Hbefore o [<- | []].
    pose proof (Hbefore _ (or_introl eq_refl)) as Hb.
    unfold equation_before in Hb. unfold equation_after.
    cbn [fst snd] in *. rewrite Hkind.
    destruct n as [x | sym]; [ exact (Hb Hn) | exact Hb ].
  - split.
    + intros t t' Hproject Hbefore n k Hin.
      exact (Htype t t' Hproject Hbefore (n, k) Hin).
    + intros extension extension' Hproject Hbefore n k Hin.
      exact (Hextension extension extension' Hproject Hbefore (n, k) Hin).
Qed.

End Project_equations_kinded.

(* ---------------------------------------------------------------------- *)
(* A projected type is well formed                                         *)
(* ---------------------------------------------------------------------- *)

Lemma traverse_keys {K A B : Type} (f : K * A -> option (K * B)) :
  (∀ (b : K * A) (b' : K * B), f b = Some b' -> fst b' = fst b) ->
  ∀ (l : list (K * A)) (l' : list (K * B)),
    traverse f l = Some l' -> map fst l' = map fst l.
Proof.
  intros Hkeys.
  induction l as [ | b l IH]; intros l' H; cbn [traverse] in H.
  - injection H as <-. reflexivity.
  - destruct (f b) as [b' | ] eqn:Hb; [ | discriminate ].
    destruct (traverse f l) as [l0 | ]; [ | discriminate ].
    injection H as <-. cbn [map]. rewrite (Hkeys b b' Hb), (IH l0 eq_refl).
    reflexivity.
Qed.

Lemma traverse_length {A B : Type} (f : A -> option B) :
  ∀ (l : list A) (l' : list B),
    traverse f l = Some l' -> length l' = length l.
Proof.
  induction l as [ | a l IH]; intros l' H; cbn [traverse] in H.
  - injection H as <-. reflexivity.
  - destruct (f a) as [b | ]; [ | discriminate ].
    destruct (traverse f l) as [l0 | ]; [ | discriminate ].
    injection H as <-. cbn [length]. rewrite (IH l0 eq_refl). reflexivity.
Qed.

Lemma project_equations_keys (to_project : list variable)
    (f : name * flambda_type -> option (list (name * flambda_type)))
    (project : flambda_type -> option flambda_type) :
  (∀ (n : name) (ty : flambda_type),
     f (n, ty)
     = if projected to_project n
       then Some []
       else
         match project ty with
         | Some ty' => Some [(n, ty')]
         | None => None
         end) ->
  ∀ (equations : list (name * flambda_type))
    (lists : list (list (name * flambda_type))),
    traverse f equations = Some lists ->
    map fst (concat lists)
    = filter (fun n => negb (projected to_project n)) (map fst equations).
Proof.
  intros Hf.
  induction equations as [ | [n ty] rest IH]; intros lists H;
    cbn [traverse] in H.
  - injection H as <-. reflexivity.
  - rewrite Hf in H. cbn [map fst filter].
    destruct (projected to_project n) eqn:Hp; cbn [negb].
    + destruct (traverse f rest) as [lists0 | ] eqn:Hrest; [ | discriminate ].
      injection H as <-. cbn [concat app]. exact (IH lists0 eq_refl).
    + destruct (project ty) as [ty' | ]; [ | discriminate ].
      destruct (traverse f rest) as [lists0 | ] eqn:Hrest; [ | discriminate ].
      injection H as <-. cbn [concat app map fst].
      rewrite (IH lists0 eq_refl). reflexivity.
Qed.

Section Project_well_formed.

Context (to_project : list variable)
  (expand : variable -> option flambda_type).

Hypothesis expand_well_formed :
  ∀ (e : variable) (ty : flambda_type),
    among e to_project = true ->
    expand e = Some ty ->
    type_well_formed ty = true.

Lemma project_descr_well_formed {head : Type}
    (project_head : head -> option head)
    (of_expanded : flambda_type -> option (type_descr head))
    (wrap : type_descr head -> flambda_type) (ty ty' : type_descr head) :
  (∀ (expanded : flambda_type) (d : type_descr head),
     of_expanded expanded = Some d -> expanded = wrap d) ->
  (∀ h h' : head,
     ty = Ok (No_alias h) ->
     project_head h = Some h' ->
     type_well_formed (wrap (Ok (No_alias h))) = true ->
     type_well_formed (wrap (Ok (No_alias h'))) = true) ->
  project_descr to_project expand project_head of_expanded ty = Some ty' ->
  type_well_formed (wrap ty) = true ->
  type_well_formed (wrap ty') = true.
Proof.
  intros Hof Hhead. unfold project_descr.
  destruct ty as [ | [h | [e | sym | c]] | ].
  - intros H. injection H as <-. exact (fun H => H).
  - destruct (project_head h) as [h' | ] eqn:Hh; [ | discriminate ].
    intros H. injection H as <-. exact (Hhead h h' eq_refl Hh).
  - destruct (among e to_project) eqn:Hproject.
    + destruct (expand e) as [expanded | ] eqn:Hexpand; [ | discriminate ].
      intros Hd _. rewrite <- (Hof _ _ Hd).
      exact (expand_well_formed e expanded Hproject Hexpand).
    + intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
Qed.

Lemma project_all_well_formed :
  (∀ t t' : flambda_type,
     project_variables_out to_project expand t = Some t' ->
     type_well_formed t = true ->
     type_well_formed t' = true)
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       env_extension_well_formed extension = true ->
       env_extension_well_formed extension' = true).
Proof.
  assert (Hall :
    (∀ t t' : flambda_type,
       project_variables_out to_project expand t = Some t' ->
       type_well_formed t = true ->
       type_well_formed t' = true)
    ∧ (∀ head head' : head_of_kind_value,
         project_head_of_kind_value to_project expand head = Some head' ->
         head_of_kind_value_well_formed head = true ->
         head_of_kind_value_well_formed head' = true)
    ∧ (∀ head head' : head_of_kind_value_non_null,
         project_head_of_kind_value_non_null to_project expand head
         = Some head' ->
         head_of_kind_value_non_null_well_formed head = true ->
         head_of_kind_value_non_null_well_formed head' = true)
    ∧ (∀ blocks blocks' : row_like_for_blocks,
         project_row_like_for_blocks to_project expand blocks = Some blocks'
         ->
         row_like_for_blocks_well_formed blocks = true ->
         row_like_for_blocks_well_formed blocks' = true)
    ∧ (∀ closures closures' : row_like_for_closures,
         project_row_like_for_closures to_project expand closures
         = Some closures' ->
         row_like_for_closures_well_formed closures = true ->
         row_like_for_closures_well_formed closures' = true)
    ∧ (∀ (entry entry' : closures_entry)
         (index : row_like_index set_of_closures_contents unit),
         project_closures_entry to_project expand entry = Some entry' ->
         closures_entry_well_formed index entry = true ->
         closures_entry_well_formed index entry' = true)
    ∧ (∀ extension extension' : env_extension,
         project_env_extension to_project expand extension = Some extension'
         ->
         env_extension_well_formed extension = true ->
         env_extension_well_formed extension' = true)).
  2:{ destruct Hall as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
      split; [ exact Htype | exact Hextension ]. }
  apply (types_induction
           (fun t =>
              ∀ t' : flambda_type,
                project_variables_out to_project expand t = Some t' ->
                type_well_formed t = true ->
                type_well_formed t' = true)
           (fun head =>
              ∀ head' : head_of_kind_value,
                project_head_of_kind_value to_project expand head
                = Some head' ->
                head_of_kind_value_well_formed head = true ->
                head_of_kind_value_well_formed head' = true)
           (fun head =>
              ∀ head' : head_of_kind_value_non_null,
                project_head_of_kind_value_non_null to_project expand head
                = Some head' ->
                head_of_kind_value_non_null_well_formed head = true ->
                head_of_kind_value_non_null_well_formed head' = true)
           (fun blocks =>
              ∀ blocks' : row_like_for_blocks,
                project_row_like_for_blocks to_project expand blocks
                = Some blocks' ->
                row_like_for_blocks_well_formed blocks = true ->
                row_like_for_blocks_well_formed blocks' = true)
           (fun closures =>
              ∀ closures' : row_like_for_closures,
                project_row_like_for_closures to_project expand closures
                = Some closures' ->
                row_like_for_closures_well_formed closures = true ->
                row_like_for_closures_well_formed closures' = true)
           (fun entry =>
              ∀ (entry' : closures_entry)
                (index : row_like_index set_of_closures_contents unit),
                project_closures_entry to_project expand entry = Some entry'
                ->
                closures_entry_well_formed index entry = true ->
                closures_entry_well_formed index entry' = true)
           (fun extension =>
              ∀ extension' : env_extension,
                project_env_extension to_project expand extension
                = Some extension' ->
                env_extension_well_formed extension = true ->
                env_extension_well_formed extension' = true)).
  - (* Type_value *)
    intros ty Hhead t' H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_value to_project expand)
                value_of_expanded ty) as [ty' | ] eqn:Hd; [ | discriminate ].
    injection H as <-.
    refine (project_descr_well_formed _ _ Type_value ty ty' _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros h h' Heq Hh. subst ty. exact (Hhead h' Hh).
  - (* Type_naked_immediate *)
    intros ty t' H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_naked_immediate to_project expand)
                naked_immediate_of_expanded ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-.
    refine (project_descr_well_formed _ _ Type_naked_immediate ty ty' _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros h h' _ Hh. unfold project_head_of_kind_naked_immediate in Hh.
      destruct (traverse _ (inverse_relations h)); [ | discriminate ].
      injection Hh as <-. exact (fun H => H).
  - (* Type_naked_number *)
    intros k ty t' H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                (naked_number_of_expanded k) ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-.
    refine (project_descr_well_formed _ _ (Type_naked_number k) ty ty' _ _
              Hd).
    + intros expanded d. exact (of_expanded_naked_number k expanded d).
    + intros h h' _ Hh. injection Hh as <-. exact (fun H => H).
  - (* Type_rec_info *)
    intros ty t' H. simpl in H.
    destruct (project_descr _ _ _ rec_info_of_expanded ty) as [ty' | ];
      [ | discriminate ].
    injection H as <-. reflexivity.
  - (* Type_region *)
    intros ty t' H. simpl in H.
    destruct (project_descr _ _ _ region_of_expanded ty) as [ty' | ];
      [ | discriminate ].
    injection H as <-. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null head' H. simpl in H.
    destruct (match is_null with
              | Not_null => Some Not_null
              | Maybe_null is_null_var =>
                let* is_null_var :=
                  project_relation to_project expand is_null_var
                in
                Some (Maybe_null is_null_var)
              end) as [is_null' | ]; [ | discriminate ].
    destruct non_null as [ | head | ].
    + injection H as <-. exact (fun H => H).
    + destruct (project_head_of_kind_value_non_null to_project expand head)
        as [head0 | ] eqn:Hhead0; [ | discriminate ].
      injection H as <-. exact (Hnon_null head0 eq_refl).
    + injection H as <-. exact (fun H => H).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions head' H.
    simpl in H.
    destruct (project_relation to_project expand is_int) as [is_int' | ];
      [ | discriminate ].
    destruct (project_relation to_project expand get_tag) as [get_tag' | ];
      [ | discriminate ].
    assert (Himm :
              ∀ immediates' : Or_unknown.t flambda_type,
                match immediates with
                | Or_unknown.Known immediates =>
                  let* immediates :=
                    project_variables_out to_project expand immediates
                  in
                  Some (Or_unknown.Known immediates)
                | Or_unknown.Unknown => Some Or_unknown.Unknown
                end
                = Some immediates' ->
                match immediates with
                | Or_unknown.Known ty =>
                  kind_eqb (kind ty) (Naked_number Kind_naked_immediate)
                  && type_well_formed ty
                | Or_unknown.Unknown => true
                end
                = true ->
                match immediates' with
                | Or_unknown.Known ty =>
                  kind_eqb (kind ty) (Naked_number Kind_naked_immediate)
                  && type_well_formed ty
                | Or_unknown.Unknown => true
                end
                = true).
    { intros immediates' Hi. destruct immediates as [imm | ].
      - destruct (project_variables_out to_project expand imm) as [imm' | ]
          eqn:Himm'; [ | discriminate ].
        injection Hi as <-. rewrite !andb_true_iff. intros [Hkind Hwf].
        split; [ | exact (Himmediates imm' eq_refl Hwf) ].
        rewrite (project_kind to_project expand imm imm' Himm'). exact Hkind.
      - injection Hi as <-. exact (fun H => H). }
    destruct (match immediates with
              | Or_unknown.Known immediates =>
                let* immediates :=
                  project_variables_out to_project expand immediates
                in
                Some (Or_unknown.Known immediates)
              | Or_unknown.Unknown => Some Or_unknown.Unknown
              end) as [immediates' | ] eqn:Himmediates'; [ | discriminate ].
    assert (Hrow :
              ∀ blocks' : Or_unknown.t row_like_for_blocks,
                match blocks with
                | Or_unknown.Known blocks =>
                  let* blocks :=
                    project_row_like_for_blocks to_project expand blocks
                  in
                  Some (Or_unknown.Known blocks)
                | Or_unknown.Unknown => Some Or_unknown.Unknown
                end
                = Some blocks' ->
                match blocks with
                | Or_unknown.Known blocks =>
                  row_like_for_blocks_well_formed blocks
                | Or_unknown.Unknown => true
                end
                = true ->
                match blocks' with
                | Or_unknown.Known blocks =>
                  row_like_for_blocks_well_formed blocks
                | Or_unknown.Unknown => true
                end
                = true).
    { intros blocks' Hb. destruct blocks as [row | ].
      - destruct (project_row_like_for_blocks to_project expand row)
          as [row' | ] eqn:Hrow'; [ | discriminate ].
        injection Hb as <-. exact (Hblocks row' eq_refl).
      - injection Hb as <-. exact (fun H => H). }
    destruct (match blocks with
              | Or_unknown.Known blocks =>
                let* blocks :=
                  project_row_like_for_blocks to_project expand blocks
                in
                Some (Or_unknown.Known blocks)
              | Or_unknown.Unknown => Some Or_unknown.Unknown
              end) as [blocks' | ] eqn:Hblocks'; [ | discriminate ].
    assert (Hext :
              ∀ extensions' : variant_extensions,
                match extensions with
                | No_extensions => Some No_extensions
                | Ext when_immediate when_block =>
                  let* when_immediate :=
                    project_env_extension to_project expand when_immediate
                  in
                  let* when_block :=
                    project_env_extension to_project expand when_block
                  in
                  Some (Ext when_immediate when_block)
                end
                = Some extensions' ->
                match extensions with
                | No_extensions => true
                | Ext when_immediate when_block =>
                  env_extension_well_formed when_immediate
                  && env_extension_well_formed when_block
                end
                = true ->
                match extensions' with
                | No_extensions => true
                | Ext when_immediate when_block =>
                  env_extension_well_formed when_immediate
                  && env_extension_well_formed when_block
                end
                = true).
    { intros extensions' He. destruct extensions as [ | wi wb].
      - injection He as <-. exact (fun H => H).
      - destruct (project_env_extension to_project expand wi) as [wi' | ]
          eqn:Hwi; [ | discriminate ].
        destruct (project_env_extension to_project expand wb) as [wb' | ]
          eqn:Hwb; [ | discriminate ].
        injection He as <-. destruct Hextensions as [IHwi IHwb].
        rewrite !andb_true_iff. intros [Hi Hb].
        split; [ exact (IHwi wi' eq_refl Hi) | exact (IHwb wb' eq_refl Hb) ]. }
    destruct (match extensions with
              | No_extensions => Some No_extensions
              | Ext when_immediate when_block =>
                let* when_immediate :=
                  project_env_extension to_project expand when_immediate
                in
                let* when_block :=
                  project_env_extension to_project expand when_block
                in
                Some (Ext when_immediate when_block)
              end) as [extensions' | ] eqn:Hextensions'; [ | discriminate ].
    injection H as <-. rewrite !variant_well_formed, !andb_true_iff.
    intros [[Hi Hb] He].
    split; [ split | ].
    + exact (Himm immediates' eq_refl Hi).
    + exact (Hrow blocks' eq_refl Hb).
    + exact (Hext extensions' eq_refl He).
  - (* Mutable_block *)
    intros alloc_mode head' H. injection H as <-. exact (fun H => H).
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents head' H. simpl in H.
    destruct (project_variables_out to_project expand contents)
      as [contents' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. rewrite !boxed_number_well_formed, !andb_true_iff.
    intros [Hkind Hwf]. split; [ | exact (Hcontents contents' eq_refl Hwf) ].
    rewrite (project_kind to_project expand contents contents' Hc).
    exact Hkind.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures head' H. simpl in H.
    destruct (project_row_like_for_closures to_project expand
                by_function_slot) as [closures' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. exact (Hclosures closures' eq_refl).
  - (* String *)
    intros strs head' H. injection H as <-. exact (fun H => H).
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother blocks' H.
    simpl in H.
    assert (Hcase :
              ∀ case case' : row_like_block_case,
                P_block_case
                  (fun t =>
                     ∀ t' : flambda_type,
                       project_variables_out to_project expand t = Some t' ->
                       type_well_formed t = true ->
                       type_well_formed t' = true)
                  (fun extension =>
                     ∀ extension' : env_extension,
                       project_env_extension to_project expand extension
                       = Some extension' ->
                       env_extension_well_formed extension = true ->
                       env_extension_well_formed extension' = true)
                  case ->
                project_block_case (project_variables_out to_project expand)
                  (project_env_extension to_project expand) case
                = Some case' ->
                row_like_block_case_well_formed case = true ->
                row_like_block_case_well_formed case' = true).
    { intros case case' [Htypes Hextension] Hp.
      unfold project_block_case in Hp.
      destruct (project_env_extension to_project expand
                  (case_env_extension case)) as [ext' | ] eqn:Hext';
        [ | discriminate ].
      destruct (traverse (project_variables_out to_project expand)
                  (case_maps_to case)) as [maps' | ] eqn:Hmaps;
        [ | discriminate ].
      injection Hp as <-. intros Hwf.
      apply (proj1 (row_like_block_case_well_formed_iff _)) in Hwf.
      apply (proj2 (row_like_block_case_well_formed_iff _)).
      cbn [case_index case_maps_to case_env_extension].
      destruct Hwf as [Hfits [Hshape [Htys Hext]]].
      split; [ | split; [ exact Hshape | split ] ].
      - unfold block_product_fits in *.
        rewrite (traverse_length _ _ _ Hmaps). exact Hfits.
      - intros ty' Hin.
        destruct (traverse_in _ _ _ Hmaps ty' Hin) as [ty [Hty Hproj]].
        exact (Htypes ty Hty ty' Hproj (Htys ty Hty)).
      - exact (Hextension ext' eq_refl Hext). }
    match type of H with
    | context [traverse ?f known_tags] =>
      destruct (traverse f known_tags) as [known_tags' | ] eqn:Htraverse;
        [ | discriminate ];
      assert (Hkeys :
                ∀ (b : Z * Or_unknown.t row_like_block_case)
                  (b' : Z * Or_unknown.t row_like_block_case),
                  f b = Some b' -> fst b' = fst b)
    end.
    { intros [k [case | ]] b' Hb; cbn [fst snd] in Hb.
      - destruct (project_block_case _ _ case) as [case' | ];
          [ | discriminate ].
        injection Hb as <-. reflexivity.
      - injection Hb as <-. reflexivity. }
    destruct (traverse_lookup Z.eqb _ Hkeys _ _ Htraverse) as [Hback _].
    assert (Hother' :
              ∃ other_tags' : Or_bottom.t row_like_block_case,
                Some (Row_like_for_blocks known_tags' other_tags' alloc_mode)
                = Some blocks'
                ∧ (match other_tags with
                   | Or_bottom.Ok case =>
                     row_like_block_case_well_formed case = true
                   | Or_bottom.Bottom => True
                   end ->
                   match other_tags' with
                   | Or_bottom.Ok case =>
                     row_like_block_case_well_formed case = true
                   | Or_bottom.Bottom => True
                   end)).
    { destruct other_tags as [case | ].
      - destruct (project_block_case _ _ case) as [case' | ] eqn:Hcase';
          [ | discriminate ].
        exists (Or_bottom.Ok case'). split; [ exact H | ].
        exact (Hcase case case' Hother Hcase').
      - exists Or_bottom.Bottom. split; [ exact H | exact (fun H => H) ]. }
    clear H. destruct Hother' as [other_tags' [H Hother']].
    injection H as <-. intros Hwf.
    apply (proj1 (row_like_for_blocks_well_formed_iff _ _ _)) in Hwf.
    apply (proj2 (row_like_for_blocks_well_formed_iff _ _ _)).
    destruct Hwf as [Hdistinct [Hcases Hothers]]. split; [ | split ].
    + unfold keys in *. rewrite (traverse_keys _ Hkeys _ _ Htraverse).
      exact Hdistinct.
    + intros tag case' Hlookup'.
      destruct (Hback tag _ Hlookup') as [key0 [a [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf. destruct a as [case | ]; [ | discriminate ].
      destruct (project_block_case _ _ case) as [case0 | ] eqn:Hcase0;
        [ | discriminate ].
      injection Hf as <-.
      exact (Hcase case case0 (Hknown key0 case Hin) Hcase0
               (Hcases tag case Hlookup)).
    + exact (Hother' Hothers).
  - (* Row_like_for_closures *)
    intros known_closures Hknown closures' H. simpl in H.
    match type of H with
    | context [traverse ?f known_closures] =>
      destruct (traverse f known_closures) as [known' | ] eqn:Htraverse;
        [ | discriminate ];
      assert (Hkeys :
                ∀ (b : function_slot * row_like_closures_case)
                  (b' : function_slot * row_like_closures_case),
                  f b = Some b' -> fst b' = fst b)
    end.
    { intros [k case] b' Hb; cbn [fst snd] in Hb.
      destruct (project_env_extension _ _ (case_env_extension case))
        as [ext' | ]; [ | discriminate ].
      destruct (project_closures_entry _ _ (case_maps_to case))
        as [maps' | ]; [ | discriminate ].
      injection Hb as <-. reflexivity. }
    destruct (traverse_lookup Pos.eqb _ Hkeys _ _ Htraverse) as [Hback _].
    injection H as <-. intros Hwf.
    apply (proj1 (row_like_for_closures_well_formed_iff _)) in Hwf.
    apply (proj2 (row_like_for_closures_well_formed_iff _)).
    destruct Hwf as [Hdistinct Hcases]. split.
    + exact (eq_ind_r (fun l => distinct l = true) Hdistinct
               (traverse_keys _ Hkeys _ _ Htraverse)).
    + intros slot case' Hlookup'.
      destruct (Hback slot case' Hlookup') as [key0 [case [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf.
      destruct (project_env_extension to_project expand
                  (case_env_extension case)) as [ext' | ] eqn:Hext';
        [ | discriminate ].
      destruct (project_closures_entry to_project expand (case_maps_to case))
        as [maps' | ] eqn:Hmaps; [ | discriminate ].
      injection Hf as <-. destruct (Hknown key0 case Hin) as [IHentry IHext].
      pose proof (Hcases slot case Hlookup) as Hwf.
      unfold row_like_closures_case_well_formed in *.
      cbn [case_index case_maps_to case_env_extension].
      rewrite !andb_true_iff in *. destruct Hwf as [[Hslot Hentry] Hext].
      split; [ split; [ exact Hslot | ] | exact (IHext ext' Hext' Hext) ].
      exact (IHentry maps' (case_index case) Hmaps Hentry).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      entry' index H.
    simpl in H.
    match type of H with
    | context [traverse ?f closure_types] =>
      destruct (traverse f closure_types) as [closure_types' | ]
        eqn:Htraverse_c; [ | discriminate ];
      assert (Hkeys_c :
                ∀ (b : function_slot * flambda_type)
                  (b' : function_slot * flambda_type),
                  f b = Some b' -> fst b' = fst b)
    end.
    { intros [k ty] b' Hb; cbn [fst snd] in Hb.
      destruct (project_variables_out to_project expand ty) as [ty' | ];
        [ | discriminate ].
      injection Hb as <-. reflexivity. }
    match type of H with
    | context [traverse ?f value_slot_types] =>
      destruct (traverse f value_slot_types) as [value_slot_types' | ]
        eqn:Htraverse_s; [ | discriminate ];
      assert (Hkeys_s :
                ∀ (b : value_slot * flambda_type)
                  (b' : value_slot * flambda_type),
                  f b = Some b' -> fst b' = fst b)
    end.
    { intros [k ty] b' Hb; cbn [fst snd] in Hb.
      destruct (project_variables_out to_project expand ty) as [ty' | ];
        [ | discriminate ].
      injection Hb as <-. reflexivity. }
    destruct (traverse_lookup Pos.eqb _ Hkeys_c _ _ Htraverse_c)
      as [Hback_c _].
    destruct (traverse_lookup Pos.eqb _ Hkeys_s _ _ Htraverse_s)
      as [Hback_s _].
    pose proof (traverse_keys _ Hkeys_c _ _ Htraverse_c) as Hsame_c.
    pose proof (traverse_keys _ Hkeys_s _ _ Htraverse_s) as Hsame_s.
    injection H as <-. intros Hwf.
    apply (proj1 (closures_entry_well_formed_iff _ _ _ _)) in Hwf.
    apply (proj2 (closures_entry_well_formed_iff _ _ _ _)).
    destruct Hwf as [Hfit [[Hd1 [Hd2 Hd3]] [Hclosure_types Hvalue_slots]]].
    split; [ | split; [ split; [ exact Hd1 | split ] | split ] ].
    + refine (eq_ind_r
                (fun l =>
                   closures_entry_slots_fit index (domain function_types)
                     (dedup l) (domain value_slot_types')
                   = true)
                _ Hsame_c).
      exact (eq_ind_r
               (fun l =>
                  closures_entry_slots_fit index (domain function_types)
                    (domain closure_types) (dedup l)
                  = true)
               Hfit Hsame_s).
    + exact (eq_ind_r (fun l => distinct l = true) Hd2 Hsame_c).
    + exact (eq_ind_r (fun l => distinct l = true) Hd3 Hsame_s).
    + intros g ty' Hlookup'.
      destruct (Hback_c g ty' Hlookup') as [key0 [ty [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf.
      destruct (project_variables_out to_project expand ty) as [ty0 | ]
        eqn:Hty; [ | discriminate ].
      injection Hf as <-.
      destruct (Hclosure_types g ty Hlookup) as [Hkind Hwf].
      split; [ | exact (Hclosures key0 ty Hin ty0 Hty Hwf) ].
      rewrite (project_kind to_project expand ty ty0 Hty). exact Hkind.
    + intros w ty' Hlookup'.
      destruct (Hback_s w ty' Hlookup') as [key0 [ty [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf.
      destruct (project_variables_out to_project expand ty) as [ty0 | ]
        eqn:Hty; [ | discriminate ].
      injection Hf as <-.
      exact (Hslots key0 ty Hin ty0 Hty (Hvalue_slots w ty Hlookup)).
  - (* Env_extension *)
    intros equations Htypes extension' H. simpl in H.
    match type of H with
    | context [traverse ?f equations] =>
      destruct (traverse f equations) as [lists | ] eqn:Htraverse;
        [ | discriminate ];
      assert (Hf :
                ∀ (n : name) (ty : flambda_type),
                  f (n, ty)
                  = if projected to_project n
                    then Some []
                    else
                      match project_variables_out to_project expand ty with
                      | Some ty' => Some [(n, ty')]
                      | None => None
                      end)
    end.
    { intros [e | sym] ty; cbn [fst snd projected];
        [ destruct (among e to_project) | ]; reflexivity. }
    injection H as <-. intros Hwf0.
    apply (proj1 (env_extension_well_formed_iff _)) in Hwf0.
    apply (proj2 (env_extension_well_formed_iff _)).
    destruct Hwf0 as [Hdistinct Hwf]. split.
    + unfold keys in *.
      rewrite (project_equations_keys to_project _ _ Hf _ _ Htraverse).
      exact (keyed_distinct_filter name_eqb name_eqb_eq _ _ Hdistinct).
    + intros n ty' Hlookup'.
      destruct (project_equations_lookup to_project _ _ Hf _ _ Htraverse n
                  ty' Hlookup')
        as [_ [ty [Hin [Hlookup Hproject]]]].
      exact (Htypes n ty Hin ty' Hproject (Hwf n ty Hlookup)).
Qed.

End Project_well_formed.

(* ---------------------------------------------------------------------- *)
(* The statements                                                          *)
(* ---------------------------------------------------------------------- *)

Theorem project_variables_out_kind : project_variables_out_kind_statement.
Proof.
  intros to_project expand t t'. exact (project_kind to_project expand t t').
Qed.

Theorem project_variables_out_scope : project_variables_out_scope_statement.
Proof.
  intros to_project expand in_target Hexpand.
  destruct (project_all_scope to_project expand in_target Hexpand)
    as [Htype Hextension].
  split.
  - intros t t' Hproject Hbefore.
    exact (Htype t t' Hproject (fun x Hx Hkept => Hbefore x Hx Hkept)).
  - intros extension extension' Hproject Hbefore.
    exact (Hextension extension extension' Hproject
             (fun x Hx Hkept => Hbefore x Hx Hkept)).
Qed.

Theorem project_variables_out_kinded : project_variables_out_kinded_statement.
Proof.
  intros to_project expand of_kind Hexpand.
  exact (project_all_kinded to_project expand of_kind Hexpand).
Qed.

Theorem project_variables_out_equations_kinded
  : project_variables_out_equations_kinded_statement.
Proof.
  intros to_project expand of_kind Hexpand.
  destruct (project_all_equations_kinded to_project expand of_kind Hexpand)
    as [Htype Hextension].
  assert (Hbefore :
            ∀ equations : list (name * flambda_kind),
              (∀ (n : name) (k : flambda_kind),
                 (n, k) ∈ equations -> name_kept to_project n -> of_kind n k)
              ->
              ∀ equation : name * flambda_kind,
                equation ∈ equations ->
                equation_before to_project of_kind equation).
  { intros equations H [n k] Hin. unfold equation_before. cbn [fst snd].
    destruct n as [e | sym].
    - intros Hkept. exact (H _ _ Hin Hkept).
    - exact (H _ _ Hin I). }
  split.
  - intros t t' Hproject H.
    exact (Htype t t' Hproject (Hbefore _ H)).
  - intros extension extension' Hproject H.
    exact (Hextension extension extension' Hproject (Hbefore _ H)).
Qed.

Theorem project_variables_out_well_formed
  : project_variables_out_well_formed_statement.
Proof.
  intros to_project expand Hexpand.
  exact (project_all_well_formed to_project expand Hexpand).
Qed.

(* ---------------------------------------------------------------------- *)
(* The types put for the variables that the last phase removes             *)
(* ---------------------------------------------------------------------- *)

Lemma get_alias_free (ty : flambda_type) (s : simple) :
  get_alias_opt ty = Some s -> free_variables ty = free_variables_of_simple s.
Proof.
  destruct ty as [d | d | k d | d | d]; destruct d as [ | [head | s'] | ];
    cbn; intros H; try discriminate; injection H as <-; reflexivity.
Qed.

Theorem expand_variable_scope : expand_variable_scope_statement.
Proof.
  intros to_expand in_target Hbase.
  induction fuel as [ | fuel IH]; intros e ty x Hexpand Hx;
    [ discriminate | ].
  cbn [Join_env.expand_variable] in Hexpand.
  destruct (lookup e to_expand) as [ty0 | ] eqn:Hlookup; [ | discriminate ].
  cbn in Hexpand.
  destruct (get_alias_opt ty0) as [[y | sym | c] | ] eqn:Halias.
  - destruct (among y (domain to_expand)) eqn:Hy.
    + exact (IH y ty x Hexpand Hx).
    + injection Hexpand as <-.
      rewrite (get_alias_free ty0 _ Halias) in Hx. destruct Hx as [<- | []].
      apply (Hbase e ty0 y Hlookup); [ | exact Hy ].
      rewrite (get_alias_free ty0 _ Halias). left. reflexivity.
  - injection Hexpand as <-. rewrite (get_alias_free ty0 _ Halias) in Hx.
    destruct Hx.
  - injection Hexpand as <-. rewrite (get_alias_free ty0 _ Halias) in Hx.
    destruct Hx.
  - destruct (project_variables_out_scope (domain to_expand)
                (Join_env.expand_variable fuel to_expand) in_target)
      as [Htype _].
    + intros e' ty' x' _ He' Hx'. exact (IH e' ty' x' He' Hx').
    + exact (Htype ty0 ty Hexpand
               (fun x' Hx' Hkept => Hbase e ty0 x' Hlookup Hx' Hkept) x Hx).
Qed.

Theorem expand_variable_well_formed : expand_variable_well_formed_statement.
Proof.
  intros to_expand Hbase.
  induction fuel as [ | fuel IH]; intros e ty Hexpand; [ discriminate | ].
  cbn [Join_env.expand_variable] in Hexpand.
  destruct (lookup e to_expand) as [ty0 | ] eqn:Hlookup; [ | discriminate ].
  cbn in Hexpand.
  destruct (get_alias_opt ty0) as [[y | sym | c] | ] eqn:Halias.
  - destruct (among y (domain to_expand)) eqn:Hy.
    + exact (IH y ty Hexpand).
    + injection Hexpand as <-. exact (Hbase e ty0 Hlookup).
  - injection Hexpand as <-. exact (Hbase e ty0 Hlookup).
  - injection Hexpand as <-. exact (Hbase e ty0 Hlookup).
  - destruct (project_variables_out_well_formed (domain to_expand)
                (Join_env.expand_variable fuel to_expand)) as [Htype _].
    + intros e' ty' _ He'. exact (IH e' ty' He').
    + exact (Htype ty0 ty Hexpand (Hbase e ty0 Hlookup)).
Qed.

(* ---------------------------------------------------------------------- *)
(* The variables of the two extensions that the last phase adds            *)
(* ---------------------------------------------------------------------- *)

Lemma is_bound_cons {A : Type} (x y : variable) (a : A) (m : finite_map A) :
  is_bound x ((y, a) :: m) = Pos.eqb x y || is_bound x m.
Proof.
  unfold is_bound, Keyed.is_bound.
  change (Keyed.lookup Pos.eqb x ((y, a) :: m)) with (lookup x ((y, a) :: m)).
  rewrite lookup_cons. destruct (Pos.eqb x y); reflexivity.
Qed.

Definition acc_env {meet_env : Type} (acc : define_acc meet_env) : meet_env :=
  fst (fst (fst (fst acc))).

Definition acc_expand {meet_env : Type} (acc : define_acc meet_env)
    : finite_map flambda_type :=
  snd (fst (fst (fst acc))).

Section Define_scope.

Context {meet_env meet_expanded_head : Type}
  (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
  (source : Typing_env.t) (s : Join_env.t)
  (relations0 : Join_env.inverse_relations) (tenv_of_aliases : Typing_env.t)
  (unique : variable -> bool).

Hypothesis s_true : state_true source s.
Hypothesis aliases_not_existential : aliases_not_made s.
Hypothesis unique_not_key :
  ∀ var : variable, unique var = true -> is_bound var relations0 = false.

Let base (x : variable) : Prop := exists_in_target s x = true.

Definition scope_invariant (acc : define_acc meet_env) : Prop :=
  let '(_, to_expand, equations, relations, _) := acc in
  (∀ (x : variable) (ty : flambda_type),
     lookup x equations = Some ty ->
     base x
     ∧ is_bound x to_expand = false
     ∧ ∀ y : variable, y ∈ free_variables ty -> base y)
  ∧ (∀ (e : variable) (ty : flambda_type),
       lookup e to_expand = Some ty ->
       is_existential s e ∧ ∀ y : variable, y ∈ free_variables ty -> base y)
  ∧ ∀ (r : variable) (names : list (Relation.t * name)),
      lookup r relations = Some names ->
      base r
      ∧ (is_bound r relations0 = true ∨ ~ is_existential s r)
      ∧ is_bound r to_expand = false
      ∧ ∀ (pair : Relation.t * name) (y : variable),
          pair ∈ names -> y ∈ free_variables_of_name (snd pair) -> base y.

Lemma alias_base (e x : variable) :
  x
  ∈ Aliases_of_existentials.aliases_of_existential_var
      (Join_env.aliases_of_existentials s) e ->
  base x ∧ ~ is_existential s x.
Proof.
  intros Hx. destruct s_true as [_ [[_ [_ [Haliases _]]] _]].
  split; [ exact (proj1 (Haliases e x Hx)) | ].
  exact (aliases_not_existential e x Hx).
Qed.

Lemma define_step_scope (acc acc' : define_acc meet_env)
    (binding : variable * flambda_kind) :
  binding
  ∈ Bindings_in_target_env.existential_variables_in_order (Join_env.bindings s)
  ->
  scope_invariant acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  scope_invariant acc'.
Proof.
  intros Hbinding Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [var var_kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hequations [Hexpand Hrelations]].
  assert (Hvar : is_existential s var).
  { apply is_bound_lookup. exists var_kind.
    exact (in_order_lookup _ _ _ Hbinding). }
  assert (Hnot_expanded :
            ∀ x : variable,
              ~ is_existential s x -> is_bound x to_expand = false).
  { intros x Hx. destruct (is_bound x to_expand) eqn:Hb; [ | reflexivity ].
    apply is_bound_lookup in Hb. destruct Hb as [ty Hty].
    exfalso. exact (Hx (proj1 (Hexpand x ty Hty))). }
  assert (Hne : ∀ x : variable, ~ is_existential s x -> Pos.eqb x var = false).
  { intros x Hx. destruct (Pos.eqb x var) eqn:Heq; [ | reflexivity ].
    apply Pos.eqb_eq in Heq. subst x. exfalso. exact (Hx Hvar). }
  destruct (Join_env.choose_alias tenv_of_aliases _) as [[canon_var | ] | ]
    eqn:Hchoose; [ | | discriminate ].
  - (* an alias in the target stands for the variable *)
    destruct (Join_env.move_equation equations var canon_var)
      as [equations1 | ] eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    pose proof (choose_alias_in _ _ _ Hchoose) as Hcanon_in.
    destruct (alias_base var canon_var Hcanon_in) as [Hcanon_base Hcanon_not].
    assert (Hequations1 :
              ∀ (x : variable) (ty : flambda_type),
                lookup x equations1 = Some ty ->
                base x
                ∧ is_bound x
                    ((var, alias_type_of var_kind (Var canon_var)) :: to_expand)
                  = false
                ∧ ∀ y : variable, y ∈ free_variables ty -> base y).
    { unfold Join_env.move_equation in Hmove.
      destruct (is_bound canon_var equations); [ discriminate | ].
      destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
      - injection Hmove as <-. intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x canon_var) eqn:Heq.
        + apply Pos.eqb_eq in Heq. subst x. injection Hx as <-.
          split; [ exact Hcanon_base | ]. split.
          * rewrite is_bound_cons, (Hne _ Hcanon_not),
              (Hnot_expanded _ Hcanon_not). reflexivity.
          * exact (proj2 (proj2 (Hequations var ty_var Hlookup_var))).
        + rewrite lookup_unbind in Hx.
          destruct (Pos.eqb x var) eqn:Hxv; [ discriminate | ].
          destruct (Hequations x ty Hx) as [Hb [Hnb Hfv]].
          split; [ exact Hb | ]. split; [ | exact Hfv ].
          rewrite is_bound_cons, Hxv, Hnb. reflexivity.
      - injection Hmove as <-. intros x ty Hx.
        destruct (Hequations x ty Hx) as [Hb [Hnb Hfv]].
        split; [ exact Hb | ]. split; [ | exact Hfv ].
        rewrite is_bound_cons, Hnb.
        destruct (Pos.eqb x var) eqn:Hxv; [ | reflexivity ].
        apply Pos.eqb_eq in Hxv. subst x. congruence. }
    split; [ | split ].
    + intros x ty Hx.
      destruct (lookup_fold_cons _ _ _ _ _ Hx) as [[Hin ->] | Hlookup].
      * apply filter_In in Hin. destruct Hin as [Hin _].
        destruct (alias_base var x Hin) as [Hb Hnot].
        split; [ exact Hb | ]. split.
        -- rewrite is_bound_cons, (Hne _ Hnot), (Hnot_expanded _ Hnot).
           reflexivity.
        -- intros y Hy. rewrite free_variables_alias_type_of in Hy.
           destruct Hy as [<- | []]. exact Hcanon_base.
      * exact (Hequations1 x ty Hlookup).
    + intros e ty He. rewrite lookup_cons in He.
      destruct (Pos.eqb e var) eqn:Heq.
      * apply Pos.eqb_eq in Heq. subst e. injection He as <-.
        split; [ exact Hvar | ]. intros y Hy.
        rewrite free_variables_alias_type_of in Hy.
        destruct Hy as [<- | []]. exact Hcanon_base.
      * exact (Hexpand e ty He).
    + intros r names Hr. unfold Join_env.move_inverse_relation in Hr.
      destruct (lookup var relations) as [names_var | ] eqn:Hlookup_var.
      * rewrite lookup_cons in Hr. destruct (Pos.eqb r canon_var) eqn:Heq.
        -- apply Pos.eqb_eq in Heq. subst r. injection Hr as <-.
           split; [ exact Hcanon_base | ].
           split; [ right; exact Hcanon_not | ]. split.
           ++ rewrite is_bound_cons, (Hne _ Hcanon_not),
                (Hnot_expanded _ Hcanon_not). reflexivity.
           ++ intros pair y Hpair Hy. apply in_app_or in Hpair.
              destruct Hpair as [Hpair | Hpair].
              ** exact (proj2 (proj2 (proj2 (Hrelations var names_var
                                               Hlookup_var)))
                          pair y Hpair Hy).
              ** destruct (lookup canon_var (unbind var relations))
                   as [existing | ] eqn:Hexisting; [ | destruct Hpair ].
                 rewrite lookup_unbind in Hexisting.
                 destruct (Pos.eqb canon_var var); [ discriminate | ].
                 exact (proj2 (proj2 (proj2 (Hrelations canon_var existing
                                               Hexisting)))
                          pair y Hpair Hy).
        -- rewrite lookup_unbind in Hr.
           destruct (Pos.eqb r var) eqn:Hrv; [ discriminate | ].
           destruct (Hrelations r names Hr) as [Hb [Htag [Hnb Hscr]]].
           split; [ exact Hb | ]. split; [ exact Htag | ].
           split; [ | exact Hscr ]. rewrite is_bound_cons, Hrv, Hnb.
           reflexivity.
      * destruct (Hrelations r names Hr) as [Hb [Htag [Hnb Hscr]]].
        split; [ exact Hb | ]. split; [ exact Htag | ].
        split; [ | exact Hscr ]. rewrite is_bound_cons, Hnb.
        destruct (Pos.eqb r var) eqn:Hrv; [ | reflexivity ].
        apply Pos.eqb_eq in Hrv. subst r. congruence.
  - destruct (unique var) eqn:Hunique.
    + (* removed *)
      injection Hstep as <-. split; [ | split ].
      * intros x ty Hx. rewrite lookup_unbind in Hx.
        destruct (Pos.eqb x var) eqn:Hxv; [ discriminate | ].
        destruct (Hequations x ty Hx) as [Hb [Hnb Hfv]].
        split; [ exact Hb | ]. split; [ | exact Hfv ].
        rewrite is_bound_cons, Hxv, Hnb. reflexivity.
      * intros e ty He. rewrite lookup_cons in He.
        destruct (Pos.eqb e var) eqn:Heq; [ | exact (Hexpand e ty He) ].
        apply Pos.eqb_eq in Heq. subst e. injection He as <-.
        split; [ exact Hvar | ].
        destruct (lookup var equations) as [ty_var | ] eqn:Hlookup_var.
        -- exact (proj2 (proj2 (Hequations var ty_var Hlookup_var))).
        -- intros y Hy. rewrite free_variables_unknown in Hy. destruct Hy.
      * intros r names Hr.
        destruct (Hrelations r names Hr) as [Hb [Htag [Hnb Hscr]]].
        split; [ exact Hb | ]. split; [ exact Htag | ].
        split; [ | exact Hscr ]. rewrite is_bound_cons, Hnb.
        destruct (Pos.eqb r var) eqn:Hrv; [ | reflexivity ].
        apply Pos.eqb_eq in Hrv. subst r. exfalso.
        destruct Htag as [Hkey | Hnot]; [ | exact (Hnot Hvar) ].
        rewrite (unique_not_key var Hunique) in Hkey. discriminate.
    + (* kept *)
      injection Hstep as <-.
      split; [ exact Hequations | split; [ exact Hexpand | exact Hrelations ] ].
Qed.

Lemma define_fold_scope (acc acc' : define_acc meet_env) :
  scope_invariant acc ->
  List.fold_left (define_step ops s tenv_of_aliases unique)
    (Bindings_in_target_env.existential_variables_in_order
       (Join_env.bindings s))
    (Some acc)
  = Some acc' ->
  scope_invariant acc'.
Proof.
  apply (fold_left_invariant (define_step ops s tenv_of_aliases unique)
           scope_invariant (fun b => eq_refl)).
  intros a a' b Hb Ha Hstep.
  exact (define_step_scope a a' b Hb Ha Hstep).
Qed.

(* The variables kept and the variables removed. *)
Lemma define_fold_kept :
  ∀ (l : list (variable * flambda_kind)) (acc acc' : define_acc meet_env),
    List.fold_left (define_step ops s tenv_of_aliases unique) l (Some acc)
    = Some acc' ->
    ∃ kept : list (variable * flambda_kind),
      (∀ binding : variable * flambda_kind, binding ∈ kept -> binding ∈ l)
      ∧ acc_env acc' = variables_defined ops (acc_env acc) kept
      ∧ (∀ e : variable,
           is_bound e (acc_expand acc) = true ->
           is_bound e (acc_expand acc') = true)
      ∧ ∀ binding : variable * flambda_kind,
          binding ∈ l ->
          is_bound (fst binding) (acc_expand acc') = true ∨ binding ∈ kept.
Proof.
  induction l as [ | binding l IH]; intros acc acc' Hfold.
  - injection Hfold as <-. exists [].
    split; [ intros b [] | ]. split; [ reflexivity | ].
    split; [ intros e He; exact He | intros b [] ].
  - cbn [List.fold_left] in Hfold.
    destruct (define_step ops s tenv_of_aliases unique (Some acc) binding)
      as [acc1 | ] eqn:Hstep.
    2:{ rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold.
        discriminate. }
    destruct (IH acc1 acc' Hfold) as [kept [Hkept [Henv [Hmono Hall]]]].
    destruct acc as [[[[menv to_expand] equations] relations] definitions].
    cbn [define_step] in Hstep.
    assert (Hremoved :
              ∀ ty : flambda_type,
                acc_env acc1 = menv ->
                acc_expand acc1 = (fst binding, ty) :: to_expand ->
                ∃ kept0 : list (variable * flambda_kind),
                  (∀ b : variable * flambda_kind,
                     b ∈ kept0 -> b ∈ binding :: l)
                  ∧ acc_env acc' = variables_defined ops menv kept0
                  ∧ (∀ e : variable,
                       is_bound e to_expand = true ->
                       is_bound e (acc_expand acc') = true)
                  ∧ ∀ b : variable * flambda_kind,
                      b ∈ binding :: l ->
                      is_bound (fst b) (acc_expand acc') = true ∨ b ∈ kept0).
    { intros ty Hmenv Hto. exists kept.
      split; [ intros b Hb; right; exact (Hkept b Hb) | ].
      split; [ rewrite Henv, Hmenv; reflexivity | ].
      assert (Hin : ∀ e : variable,
                is_bound e ((fst binding, ty) :: to_expand) = true ->
                is_bound e (acc_expand acc') = true).
      { intros e He. apply Hmono. rewrite Hto. exact He. }
      split.
      - intros e He. apply Hin. rewrite is_bound_cons, He.
        apply orb_true_r.
      - intros b [<- | Hb]; [ | exact (Hall b Hb) ].
        left. apply Hin. rewrite is_bound_cons, Pos.eqb_refl. reflexivity. }
    destruct (Join_env.choose_alias _ _) as [[canon_var | ] | ];
      [ | | discriminate ].
    + destruct (Join_env.move_equation equations (fst binding) canon_var);
        [ | discriminate ].
      injection Hstep as <-. exact (Hremoved _ eq_refl eq_refl).
    + destruct (unique (fst binding)).
      * injection Hstep as <-. exact (Hremoved _ eq_refl eq_refl).
      * injection Hstep as <-. exists (binding :: kept). split.
        -- intros b [<- | Hb];
             [ left; reflexivity | right; exact (Hkept b Hb) ].
        -- split; [ rewrite Henv; destruct binding; reflexivity | ].
           split; [ exact Hmono | ].
           intros b [<- | Hb]; [ right; left; reflexivity | ].
           destruct (Hall b Hb) as [Hbound | Hin];
             [ left; exact Hbound | right; right; exact Hin ].
Qed.

End Define_scope.

Lemma naked_immediate_relations_free (names : list (Relation.t * name))
    (y : variable) :
  y
  ∈ free_variables
      (Type_naked_immediate
         (Ok
            (No_alias
               (Head_of_kind_naked_immediate Or_unknown.Unknown names))))
  ->
  ∃ pair : Relation.t * name,
    pair ∈ names ∧ y ∈ free_variables_of_name (snd pair).
Proof.
  cbn. intros Hy. apply in_flat_map in Hy. exact Hy.
Qed.

(* A variable of the target so far that the last phase does not remove
   is one of the target that it answers. *)
Lemma base_in_final_target (s : Join_env.t)
    (to_expand : finite_map flambda_type)
    (kept : list (variable * flambda_kind)) :
  (∀ binding : variable * flambda_kind,
     binding
     ∈ Bindings_in_target_env.existential_variables_in_order
         (Join_env.bindings s) ->
     is_bound (fst binding) to_expand = true ∨ binding ∈ kept) ->
  ∀ y : variable,
    exists_in_target s y = true ->
    is_bound y to_expand = false ->
    in_final_target s kept y.
Proof.
  intros Hall y Hy Hnot. unfold exists_in_target_env, kind_in_target_env in Hy.
  unfold in_final_target.
  destruct
    (Keyed.lookup name_eqb (Name_var y)
       (Typing_env.names_to_types
          (Bindings_in_target_env.source_env (Join_env.bindings s))))
    as [[ty btm] | ] eqn:Hsource.
  - left. apply bound_knows. unfold Keyed.is_bound, Join_env.source_env.
    rewrite Hsource. reflexivity.
  - destruct (lookup y (Bindings_in_target_env.existential_variables
                          (Join_env.bindings s))) as [k | ] eqn:Hexistential.
    + right. right.
      assert (Hin : (y, k)
                    ∈ Bindings_in_target_env.existential_variables_in_order
                        (Join_env.bindings s)).
      { unfold Bindings_in_target_env.existential_variables_in_order.
        apply in_rev. rewrite rev_involutive. apply in_compact.
        exact Hexistential. }
      destruct (Hall (y, k) Hin) as [Hbound | Hkept].
      * cbn [fst] in Hbound. rewrite Hbound in Hnot. discriminate.
      * apply in_map_iff. exists (y, k). split; [ reflexivity | exact Hkept ].
    + right. left. unfold is_imported. apply is_bound_lookup.
      destruct (lookup y (Bindings_in_target_env.imported_variables
                            (Join_env.bindings s))) as [k | ];
        [ exists k; reflexivity | discriminate ].
Qed.

Lemma unique_not_key (relations : Join_env.inverse_relations) (b : bool)
    (var : variable) :
  negb (Join_env.in_inverse_relations relations var) && b = true ->
  is_bound var relations = false.
Proof.
  unfold Join_env.in_inverse_relations.
  destruct (is_bound var relations); [ discriminate | reflexivity ].
Qed.

Theorem define_variables_in_scope : define_variables_in_scope_statement.
Proof.
  intros meet_env meet_expanded_head ops fuel mh source s menv target
    relations definitions Htrue Hrelations Hnonempty Htarget Hnot_made
    Hdefine.
  unfold Join_env.define_or_eliminate_variables_and_add_equations in Hdefine.
  cbv zeta in Hdefine.
  match type of Hdefine with
  | context [List.fold_left ?f
               (Bindings_in_target_env.existential_variables_in_order ?b)
               (Some ?init)] =>
    destruct (List.fold_left f
                (Bindings_in_target_env.existential_variables_in_order b)
                (Some init))
      as [[[[[menv1 to_expand] equations] relations1] definitions1] | ]
      eqn:Hfold; [ | discriminate ];
    set (start := init) in Hfold
  end.
  match type of Hdefine with
  | context [traverse ?f (compact equations)] =>
    destruct (traverse f (compact equations)) as [equations' | ]
      eqn:Hequations; [ | discriminate ]
  end.
  match type of Hdefine with
  | context [traverse ?f (compact relations1)] =>
    destruct (traverse f (compact relations1)) as [relation_equations | ]
      eqn:Hrelation_equations; [ | discriminate ]
  end.
  destruct (Join_env.add_env_extension ops menv1 (Env_extension equations') mh)
    as [menv2 | ] eqn:Hadd1; [ | discriminate ].
  destruct (Join_env.add_env_extension ops menv2
              (Env_extension relation_equations) mh) as [menv3 | ] eqn:Hadd2;
    [ | discriminate ].
  injection Hdefine as <- <-.
  set (unique :=
         fun var : variable =>
           negb (Join_env.in_inverse_relations relations var)
           && Nat.leb
                (Join_env.count_occurrences var
                   (flat_map
                      (fun binding : variable * flambda_type =>
                         free_variables (snd binding))
                      (compact (Join_env.types_in_target_env s))))
                1).
  destruct (define_fold_kept ops s _ unique _ _ _ Hfold)
    as [kept [Hkept [Hmenv1 [_ Hall]]]].
  unfold start, acc_env in Hmenv1. cbn [fst] in Hmenv1.
  exists kept, equations', relation_equations, menv2.
  split; [ exact Hkept | ]. split.
  { unfold variables_defined. rewrite fold_left_app.
    unfold variables_defined in Hmenv1. rewrite <- Hmenv1. exact Hadd1. }
  split; [ exact Hadd2 | ].
  assert (Hstart : scope_invariant s relations start).
  { unfold start, scope_invariant. split; [ | split ].
    - intros x ty Hlookup.
      destruct Htrue as [_ [[Hhold _] _]].
      split; [ exact (proj1 (Hhold x ty Hlookup)) | ].
      split; [ reflexivity | ]. exact (Htarget x ty Hlookup).
    - intros e ty He. discriminate.
    - intros r names Hr.
      assert (Hpairs :
                ∀ pair : Relation.t * name,
                  pair ∈ names ->
                  ∃ x : variable,
                    snd pair = Name_var x
                    ∧ relation_holds_at s r (fst pair) x).
      { intros [relation n] Hpair.
        exact (Hrelations r relation n (ex_intro _ names (conj Hr Hpair))). }
      split.
      + destruct names as [ | pair names]; [ exfalso | ].
        * exact (Hnonempty r [] Hr eq_refl).
        * destruct (Hpairs pair (or_introl eq_refl)) as [x [_ Hholds]].
          exact (proj1 (proj2 Hholds)).
      + split.
        * left. apply is_bound_lookup. exists names. exact Hr.
        * split; [ reflexivity | ]. intros pair y Hpair Hy.
          destruct (Hpairs pair Hpair) as [x [Hname Hholds]].
          rewrite Hname in Hy. destruct Hy as [<- | []].
          exact (proj1 Hholds). }
  pose proof (define_fold_scope ops source s relations _ unique Htrue
                Hnot_made
                (fun var Hvar => unique_not_key relations _ var Hvar)
                _ _ Hstart Hfold) as Hinvariant.
  destruct Hinvariant as [Hequations_inv [Hexpand_inv Hrelations_inv]].
  pose proof (base_in_final_target s to_expand kept Hall) as Hfinal.
  assert (Hexpanded :
            ∀ (e : variable) (ty : flambda_type) (x : variable),
              among e (domain to_expand) = true ->
              Join_env.expand_variable fuel to_expand e = Some ty ->
              x ∈ free_variables ty ->
              in_final_target s kept x).
  { intros e ty x _ He Hx.
    refine (expand_variable_scope to_expand (in_final_target s kept) _ fuel e
              ty x He Hx).
    intros e0 ty0 x0 He0 Hx0 Hamong. rewrite among_domain in Hamong.
    exact (Hfinal x0 (proj2 (Hexpand_inv e0 ty0 He0) x0 Hx0) Hamong). }
  destruct (project_variables_out_scope (domain to_expand)
              (Join_env.expand_variable fuel to_expand)
              (in_final_target s kept) Hexpanded) as [Hproject _].
  split.
  - intros n ty' Hin.
    destruct (traverse_in _ _ _ Hequations (n, ty') Hin)
      as [[x ty] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    destruct (project_variables_out (domain to_expand)
                (Join_env.expand_variable fuel to_expand) ty) as [ty0 | ]
      eqn:Hty; [ | discriminate ].
    injection Hf as <- <-. apply in_compact in Hbinding.
    destruct (Hequations_inv x ty Hbinding) as [Hx [Hnot Hfree]]. split.
    + intros y [<- | []]. exact (Hfinal x Hx Hnot).
    + refine (Hproject ty ty0 Hty _). intros y Hy Hamong.
      rewrite among_domain in Hamong. exact (Hfinal y (Hfree y Hy) Hamong).
  - intros n ty' Hin.
    destruct (traverse_in _ _ _ Hrelation_equations (n, ty') Hin)
      as [[r names] [Hbinding Hf]].
    cbn [fst snd] in Hf.
    match type of Hf with
    | context [project_variables_out ?to_project ?expand ?ty] =>
      destruct (project_variables_out to_project expand ty) as [ty0 | ]
        eqn:Hty; [ | discriminate ]
    end.
    injection Hf as <- <-. apply in_compact in Hbinding.
    destruct (Hrelations_inv r names Hbinding) as [Hr [_ [Hnot Hscrutinees]]].
    split.
    + intros y [<- | []]. exact (Hfinal r Hr Hnot).
    + refine (Hproject _ ty0 Hty _). intros y Hy Hamong.
      rewrite among_domain in Hamong.
      destruct (naked_immediate_relations_free names y Hy)
        as [pair [Hpair Hy']].
      exact (Hfinal y (Hscrutinees pair y Hpair Hy') Hamong).
Qed.

(* ---------------------------------------------------------------------- *)
(* What every function of the join of environments keeps                   *)
(* ---------------------------------------------------------------------- *)

Definition scoped_but (pending : variable -> Prop) (s : Join_env.t) : Prop :=
  (∀ (x : variable) (ty : flambda_type),
     lookup x (Join_env.types_in_target_env s) = Some ty ->
     ∀ y : variable,
       y ∈ free_variables ty -> exists_in_target s y = true ∨ pending y)
  ∧ aliases_from_envs s.

Definition scope_grows (s s' : Join_env.t) : Prop :=
  (imports_listed s -> imports_listed s')
  ∧ Join_env.source_env s' = Join_env.source_env s
  ∧ (∀ x : variable,
       exists_in_target s x = true -> exists_in_target s' x = true)
  ∧ ∀ x : variable, is_imported s x -> is_imported s' x.

Definition keeps_scope (s s' : Join_env.t) : Prop :=
  scope_grows s s'
  ∧ (imports_listed s ->
     ∀ pending : variable -> Prop,
       scoped_but pending s -> scoped_but pending s').

Lemma scope_grows_refl (s : Join_env.t) : scope_grows s s.
Proof.
  split; [ intros H; exact H | ]. split; [ reflexivity | ].
  split; intros x H; exact H.
Qed.

Lemma scope_grows_trans (s1 s2 s3 : Join_env.t) :
  scope_grows s1 s2 -> scope_grows s2 s3 -> scope_grows s1 s3.
Proof.
  intros [Hl [Hs [He Hi]]] [Hl' [Hs' [He' Hi']]].
  split; [ intros H; exact (Hl' (Hl H)) | ].
  split; [ rewrite Hs'; exact Hs | ].
  split; intros x H; [ exact (He' x (He x H)) | exact (Hi' x (Hi x H)) ].
Qed.

Lemma keeps_scope_refl (s : Join_env.t) : keeps_scope s s.
Proof.
  split; [ apply scope_grows_refl | intros _ pending H; exact H ].
Qed.

Lemma keeps_scope_trans (s1 s2 s3 : Join_env.t) :
  keeps_scope s1 s2 -> keeps_scope s2 s3 -> keeps_scope s1 s3.
Proof.
  intros [Hgrows Hkeeps] [Hgrows' Hkeeps'].
  split; [ exact (scope_grows_trans _ _ _ Hgrows Hgrows') | ].
  intros Hl pending H.
  exact (Hkeeps' (proj1 Hgrows Hl) pending (Hkeeps Hl pending H)).
Qed.

Lemma scoped_but_weaken (pending pending' : variable -> Prop)
    (s : Join_env.t) :
  (∀ y : variable, pending y -> exists_in_target s y = true ∨ pending' y) ->
  scoped_but pending s ->
  scoped_but pending' s.
Proof.
  intros Hweaken [Hequations Hfrom]. split; [ | exact Hfrom ].
  intros x ty Hx y Hy.
  destruct (Hequations x ty Hx y Hy) as [H | H];
    [ left; exact H | exact (Hweaken y H) ].
Qed.

Lemma scoped_but_nothing (s : Join_env.t) :
  state_scoped s <-> scoped_but (fun _ => False) s.
Proof.
  split; intros [Hequations Hfrom]; (split; [ | exact Hfrom ]).
  - intros x ty Hx y Hy. left. exact (Hequations x ty Hx y Hy).
  - intros x ty Hx y Hy.
    destruct (Hequations x ty Hx y Hy) as [H | []]. exact H.
Qed.

Lemma imported_exists (s : Join_env.t) (x : variable) :
  is_imported s x -> exists_in_target s x = true.
Proof.
  intros Hx. rewrite kind_in_target_env_exists.
  unfold Bindings_in_target_env.exists_in_target_env.
  unfold is_imported in Hx. rewrite Hx. apply orb_true_r.
Qed.

(* A state whose equations and aliases are those of another, and that
   has grown from it. *)
Lemma same_maps_keeps_scope (s s' : Join_env.t) :
  scope_grows s s' ->
  Join_env.types_in_target_env s' = Join_env.types_in_target_env s ->
  Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s ->
  keeps_scope s s'.
Proof.
  intros Hgrows Htypes Haliases. split; [ exact Hgrows | ].
  destruct Hgrows as [_ [Hsource [Hexists Himported]]].
  intros _ pending [Hequations Hfrom]. split.
  - intros x ty Hx y Hy. rewrite Htypes in Hx.
    destruct (Hequations x ty Hx y Hy) as [H | H];
      [ left; exact (Hexists y H) | right; exact H ].
  - intros e x Hx. rewrite Haliases in Hx. rewrite Hsource.
    destruct (Hfrom e x Hx) as [H | H];
      [ left; exact H | right; exact (Himported x H) ].
Qed.

Lemma same_readings_scope_grows (s s' : Join_env.t) :
  same_readings s s' -> scope_grows s s'.
Proof.
  destruct s, s'. unfold same_readings. cbn.
  intros [-> [-> [-> ->]]].
  split; [ intros H; exact H | ]. split; [ reflexivity | ].
  split; intros x H; exact H.
Qed.

Lemma existential_scope_grows (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  scope_grows s s'
  ∧ Join_env.types_in_target_env s' = Join_env.types_in_target_env s
  ∧ Join_env.aliases_of_existentials s' = Join_env.aliases_of_existentials s.
Proof.
  unfold Join_env.existential_for_these_canonicals. intros Hmake.
  destruct (Bindings_in_target_env.existential_for_these_simples
              (Join_env.fresh p) (Join_env.bindings s) canonicals kind)
    as [x0 bindings0] eqn:Hbindings.
  injection Hmake as _ <-.
  split; [ | split; reflexivity ].
  unfold Bindings_in_target_env.existential_for_these_simples in Hbindings.
  destruct (Bindings_in_target_env.has_existential_for_these_simples
              (Join_env.bindings s) canonicals).
  - injection Hbindings as _ <-.
    split; [ intros H; exact H | ]. split; [ reflexivity | ].
    split; intros y H; exact H.
  - injection Hbindings as _ <-.
    split; [ intros H; exact H | ]. split; [ reflexivity | ].
    split; [ | intros y H; exact H ].
    intros y. rewrite !kind_in_target_env_exists.
    unfold Bindings_in_target_env.exists_in_target_env.
    cbn [Join_env.bindings Bindings_in_target_env.source_env
           Bindings_in_target_env.existential_variables
           Bindings_in_target_env.imported_variables].
    rewrite is_bound_cons.
    destruct (Source_env.exists_in_source_env _ y); [ reflexivity | ].
    destruct (is_bound y
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s)));
      [ intros _; rewrite orb_true_r; reflexivity | ].
    destruct (is_bound y
                (Bindings_in_target_env.imported_variables
                   (Join_env.bindings s)));
      [ intros _; apply orb_true_r | discriminate ].
Qed.

Lemma existential_keeps_scope (p : Join_env.parameters) (s s' : Join_env.t)
    (canonicals : Simples_in_joined_envs.t) (kind : flambda_kind)
    (x : variable) :
  Join_env.existential_for_these_canonicals p s canonicals kind = (x, s') ->
  keeps_scope s s'.
Proof.
  intros Hmake.
  destruct (existential_scope_grows _ _ _ _ _ _ Hmake)
    as [Hgrows [Htypes Haliases]].
  exact (same_maps_keeps_scope _ _ Hgrows Htypes Haliases).
Qed.

Lemma add_equation_scope (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  scope_grows s s'
  ∧ ∀ pending : variable -> Prop,
      (∀ y : variable,
         y ∈ free_variables ty -> exists_in_target s y = true ∨ pending y) ->
      scoped_but pending s ->
      scoped_but pending s'.
Proof.
  intros Hadd.
  pose proof (add_equation_same_readings _ _ _ _ Hadd) as Hsame.
  split; [ exact (same_readings_scope_grows _ _ Hsame) | ].
  intros pending Hty [Hequations Hfrom].
  unfold Join_env.add_equation_in_target_env in Hadd.
  destruct (is_bound x (Join_env.types_in_target_env s)); [ discriminate | ].
  injection Hadd as <-. split; [ | exact Hfrom ].
  intros x0 ty0 Hx0 y Hy. cbn [Join_env.types_in_target_env] in Hx0.
  rewrite lookup_cons in Hx0. destruct (Pos.eqb x0 x).
  - injection Hx0 as <-. exact (Hty y Hy).
  - exact (Hequations x0 ty0 Hx0 y Hy).
Qed.

Lemma add_equation_keeps_scope (s s' : Join_env.t) (x : variable)
    (ty : flambda_type) :
  Join_env.add_equation_in_target_env s x ty = Some s' ->
  (imports_listed s ->
   ∀ y : variable, y ∈ free_variables ty -> exists_in_target s y = true) ->
  keeps_scope s s'.
Proof.
  intros Hadd Hty.
  destruct (add_equation_scope _ _ _ _ Hadd) as [Hgrows Hkeeps].
  split; [ exact Hgrows | ]. intros Hl pending Hscoped.
  apply (Hkeeps pending); [ | exact Hscoped ].
  intros y Hy. left. exact (Hty Hl y Hy).
Qed.

Lemma add_alias_keeps_scope (p : Join_env.parameters) (s : Join_env.t)
    (kind : flambda_kind) (var : variable)
    (canonicals : Simples_in_joined_envs.t) :
  knows (Join_env.source_env s) var ∨ is_imported s var ->
  keeps_scope s (Join_env.add_alias_of_existential p s kind var canonicals).
Proof.
  intros Hvar. unfold Join_env.add_alias_of_existential.
  destruct (Join_env.existential_for_these_canonicals p s canonicals kind)
    as [x s1] eqn:Hmake.
  destruct (existential_keeps_scope _ _ _ _ _ _ Hmake) as [Hgrows Hkeeps].
  split; [ exact Hgrows | ].
  intros Hl pending Hscoped.
  destruct (Hkeeps Hl pending Hscoped) as [Hequations Hfrom].
  split; [ exact Hequations | ].
  intros e y Hy. cbn [Join_env.aliases_of_existentials] in Hy.
  apply aliases_add_cases in Hy. destruct Hy as [Hy | [_ ->]].
  - exact (Hfrom e y Hy).
  - destruct Hgrows as [_ [Hsource [_ Himported]]].
    destruct Hvar as [H | H].
    + left. unfold Join_env.source_env in Hsource |- *.
      cbn [Join_env.bindings]. rewrite Hsource. exact H.
    + right. exact (Himported var H).
Qed.

Definition import_keeps_scope
    (import_var : Join_env.t -> variable -> option Join_env.t) : Prop :=
  ∀ (s s' : Join_env.t) (x : variable),
    import_var s x = Some s' ->
    keeps_scope s s' ∧ (imports_listed s -> exists_in_target s' x = true).

Lemma import_vars_keeps_scope
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (vars : list variable) :
  import_keeps_scope import_var ->
  ∀ s s' : Join_env.t,
    List.fold_left
      (fun env var =>
         let* env := env in
         import_var env var)
      vars (Some s)
    = Some s' ->
    keeps_scope s s'
    ∧ (imports_listed s ->
       ∀ x : variable, x ∈ vars -> exists_in_target s' x = true).
Proof.
  intros Himport.
  induction vars as [ | var vars IH]; intros s s' Hfold.
  - injection Hfold as <-.
    split; [ apply keeps_scope_refl | intros _ x [] ].
  - cbn [List.fold_left] in Hfold.
    destruct (import_var s var) as [s1 | ] eqn:Hvar.
    + destruct (Himport _ _ _ Hvar) as [Hkeeps1 Hexists1].
      destruct (IH _ _ Hfold) as [Hkeeps Hall].
      split; [ exact (keeps_scope_trans _ _ _ Hkeeps1 Hkeeps) | ].
      intros Hl x [<- | Hx].
      * exact (proj1 (proj2 (proj2 (proj1 Hkeeps))) _ (Hexists1 Hl)).
      * exact (Hall (proj1 (proj1 Hkeeps1) Hl) x Hx).
    + exfalso. clear -Hfold.
      induction vars as [ | var' vars IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma add_equals_keeps_scope (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (kind : flambda_kind) (var : variable)
    (simples : Simples_in_joined_envs.t) :
  import_keeps_scope import_var ->
  knows (Join_env.source_env s) var ∨ is_imported s var ->
  Join_env.add_equals_in_joined_envs_with p import_var s kind var simples
  = Some s' ->
  keeps_scope s s'.
Proof.
  intros Himport Hvar Hadd.
  unfold Join_env.add_equals_in_joined_envs_with in Hadd.
  destruct (Join_env.n_way_join_canonical_simples s simples)
    as [[canonical | imported | canonicals] | ] eqn:Hcanonical;
    [ | | | discriminate ].
  - destruct (canonical_simples_cases s simples _ Hcanonical) as [_ Hsource].
    apply (add_equation_keeps_scope _ _ _ _ Hadd).
    intros _ y Hy. rewrite free_variables_alias_type_of in Hy.
    exact (source_exists s y (Hsource y Hy)).
  - destruct (import_var s imported) as [s1 | ] eqn:Hi; [ | discriminate ].
    destruct (Himport _ _ _ Hi) as [[Hgrows1 Hkeeps1] Hexists1].
    destruct (add_equation_scope _ _ _ _ Hadd) as [Hgrows2 Hkeeps2].
    split; [ exact (scope_grows_trans _ _ _ Hgrows1 Hgrows2) | ].
    intros Hl pending Hscoped.
    apply (Hkeeps2 pending); [ | exact (Hkeeps1 Hl pending Hscoped) ].
    intros y Hy. rewrite free_variables_alias_type_of in Hy.
    destruct Hy as [<- | []]. left. exact (Hexists1 Hl).
  - injection Hadd as <-. apply add_alias_keeps_scope. exact Hvar.
Qed.

Lemma concrete_keeps_scope (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  import_keeps_scope import_var ->
  Join_env.import_or_delay_n_way_join_of_concrete_type_with p import_var s var
    types
  = Some s' ->
  keeps_scope s s'.
Proof.
  intros Himport Hconcrete.
  unfold Join_env.import_or_delay_n_way_join_of_concrete_type_with
    in Hconcrete.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct (forallb _ _).
  - destruct (Join_env.add_equation_in_target_env s var any_ty)
      as [s1 | ] eqn:Hadd; [ | discriminate ].
    destruct (add_equation_scope _ _ _ _ Hadd) as [Hgrows1 Hscoped1].
    destruct (import_vars_keeps_scope _ _ Himport _ _ Hconcrete)
      as [[Hgrows Hkeeps] Hall].
    split; [ exact (scope_grows_trans _ _ _ Hgrows1 Hgrows) | ].
    intros Hl pending Hscoped.
    pose proof (proj1 Hgrows1 Hl) as Hl1.
    apply (scoped_but_weaken
             (fun y => pending y ∨ y ∈ free_variables any_ty) pending).
    + intros y [Hy | Hy]; [ right; exact Hy | left; exact (Hall Hl1 y Hy) ].
    + apply (Hkeeps Hl1). apply Hscoped1.
      * intros y Hy. right. right. exact Hy.
      * apply (scoped_but_weaken pending _ s); [ | exact Hscoped ].
        intros y Hy. right. left. exact Hy.
  - injection Hconcrete as <-.
    apply same_maps_keeps_scope; [ | reflexivity | reflexivity ].
    apply same_readings_scope_grows. apply add_equations_same_readings.
Qed.

Lemma with_import_keeps_scope (s : Join_env.t) (x : variable)
    (kind : flambda_kind) :
  keeps_scope s (with_import s x kind) ∧ is_imported (with_import s x kind) x.
Proof.
  assert (Hbindings :
            Bindings_in_target_env.source_env
              (Bindings_in_target_env.import_from_all_envs (Join_env.bindings s)
                 x kind)
            = Bindings_in_target_env.source_env (Join_env.bindings s)
            ∧ Bindings_in_target_env.existential_variables
                (Bindings_in_target_env.import_from_all_envs
                   (Join_env.bindings s) x kind)
              = Bindings_in_target_env.existential_variables
                  (Join_env.bindings s)
            ∧ is_bound x
                (Bindings_in_target_env.imported_variables
                   (Bindings_in_target_env.import_from_all_envs
                      (Join_env.bindings s) x kind))
              = true
            ∧ ∀ y : variable,
                is_bound y
                  (Bindings_in_target_env.imported_variables
                     (Join_env.bindings s))
                = true ->
                is_bound y
                  (Bindings_in_target_env.imported_variables
                     (Bindings_in_target_env.import_from_all_envs
                        (Join_env.bindings s) x kind))
                = true).
  { unfold Bindings_in_target_env.import_from_all_envs.
    destruct (is_bound x
                (Bindings_in_target_env.imported_variables
                   (Join_env.bindings s))) eqn:Hbound.
    - split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ exact Hbound | intros y H; exact H ].
    - split; [ reflexivity | ]. split; [ reflexivity | ].
      cbn [Bindings_in_target_env.imported_variables].
      split; [ rewrite is_bound_cons, Pos.eqb_refl; reflexivity | ].
      intros y H. rewrite is_bound_cons, H. apply orb_true_r. }
  destruct Hbindings as [Hsource [Hexistential [Hx Himported]]].
  split; [ | exact Hx ].
  apply same_maps_keeps_scope; [ | reflexivity | reflexivity ].
  split; [ | split; [ exact Hsource | split; [ | exact Himported ] ] ].
  - intros Hl y Hy. cbn [with_import Join_env.imported_vars] in Hy.
    rewrite among_cons in Hy. destruct (Pos.eqb y x) eqn:Heq.
    + apply Pos.eqb_eq in Heq. subst y. exact Hx.
    + exact (Himported y (Hl y Hy)).
  - intros y. rewrite !kind_in_target_env_exists.
    unfold Bindings_in_target_env.exists_in_target_env.
    cbn [with_import Join_env.bindings]. rewrite Hsource, Hexistential.
    destruct (Source_env.exists_in_source_env _ y); [ reflexivity | ].
    destruct (is_bound y
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))); [ reflexivity | ].
    cbn [orb]. exact (Himported y).
Qed.

Lemma import_body_keeps_scope (p : Join_env.parameters)
    (import_var : Join_env.t -> variable -> option Join_env.t) :
  import_keeps_scope import_var ->
  import_keeps_scope (Join_env.import_var_transitive_body p import_var).
Proof.
  intros Himport s s' x Hbody.
  unfold Join_env.import_var_transitive_body in Hbody.
  destruct (Source_env.exists_in_source_env (Join_env.source_env s) x)
    eqn:Hsource.
  { injection Hbody as <-.
    split; [ apply keeps_scope_refl | ].
    intros _. exact (source_exists s x Hsource). }
  destruct (among x (Join_env.imported_vars s)) eqn:Hamong.
  { injection Hbody as <-. split; [ apply keeps_scope_refl | ].
    intros Hl. exact (imported_exists s x (Hl x Hamong)). }
  destruct (Joined_envs.kind_of_imported_var (Join_env.joined_envs s) x)
    as [kind | ]; [ | discriminate ].
  cbv zeta in Hbody. fold (with_import s x kind) in Hbody.
  destruct (with_import_keeps_scope s x kind) as [Hfirst Hx].
  assert (Hrest : keeps_scope (with_import s x kind) s').
  { destruct (Joined_envs.find_imported_var _ x kind) as [types | ];
      [ | discriminate ].
    destruct (existsb _ _).
    - destruct
        (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
           _ _) as [[canonical_simples | ] | ]; try discriminate.
      exact (add_equals_keeps_scope _ _ _ _ _ _ _ Himport (or_intror Hx)
               Hbody).
    - exact (concrete_keeps_scope _ _ _ _ _ _ Himport Hbody). }
  split; [ exact (keeps_scope_trans _ _ _ Hfirst Hrest) | ].
  intros _.
  exact (proj1 (proj2 (proj2 (proj1 Hrest))) x (imported_exists _ x Hx)).
Qed.

Lemma import_var_keeps_scope (p : Join_env.parameters) (fuel : nat) :
  import_keeps_scope (Join_env.import_var_transitive p fuel).
Proof.
  induction fuel as [ | fuel IH].
  - intros s s' x H. discriminate.
  - exact (import_body_keeps_scope p _ IH).
Qed.

Lemma import_type_keeps_scope (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (ty : flambda_type) :
  Join_env.import_type_transitive p fuel s ty = Some s' ->
  keeps_scope s s'
  ∧ (imports_listed s ->
     ∀ x : variable, x ∈ free_variables ty -> exists_in_target s' x = true).
Proof.
  intros Himport.
  exact (import_vars_keeps_scope _ (free_variables ty)
           (import_var_keeps_scope p fuel) s s' Himport).
Qed.

Lemma n_way_join_simples_keeps_scope (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (kind : flambda_kind)
    (simples : list (Index.t * simple)) (result : Or_bottom.t simple) :
  Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
  keeps_scope s s'.
Proof.
  intros Hjoin. unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. apply keeps_scope_refl. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. apply keeps_scope_refl.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (proj1 (import_var_keeps_scope p fuel _ _ _ Himport)).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_keeps_scope _ _ _ _ _ _ Hmake).
Qed.

Lemma n_way_join_env_extension_keeps_scope (p : Join_env.parameters)
    (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension) :
  Join_env.n_way_join_env_extension p fuel s extensions
  = Some (Or_bottom.Ok (joined, s')) ->
  keeps_scope s s'.
Proof.
  intros Hjoin. unfold Join_env.n_way_join_env_extension in Hjoin.
  destruct extensions as [ | [index0 first] others]; [ discriminate | ].
  destruct (forallb _ others).
  - destruct (List.fold_left _ (Join_env.equations_of first) (Some s))
      as [s1 | ] eqn:Hfold; [ | discriminate ].
    injection Hjoin as _ <-. revert s Hfold.
    generalize (Join_env.equations_of first) as equations.
    induction equations as [ | [n ty] equations IH]; intros s Hfold.
    + injection Hfold as <-. apply keeps_scope_refl.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : keeps_scope s s2).
        { destruct n as [x | sym].
          - exact (proj1 (import_var_keeps_scope p fuel _ _ _ Hname)).
          - injection Hname as <-. apply keeps_scope_refl. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (keeps_scope_trans _ _ _ Hfirst
                    (keeps_scope_trans _ _ _
                       (proj1 (import_type_keeps_scope p fuel _ _ _ Htype))
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. apply keeps_scope_refl.
Qed.

Lemma import_or_delay_keeps_scope (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  knows (Join_env.source_env s) var ∨ is_imported s var ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  keeps_scope s s'.
Proof.
  intros Hvar H.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in H.
  destruct (Index.choose types) as [[index any_ty] | ]; [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn _ _)
    as [[canonical_simples | ] | ]; [ | | discriminate ].
  - exact (add_equals_keeps_scope _ _ _ _ _ _ _
             (import_var_keeps_scope p fuel) Hvar H).
  - exact (concrete_keeps_scope _ _ _ _ _ _ (import_var_keeps_scope p fuel)
             H).
Qed.

Lemma join_aliases_fold_keeps_scope (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (test : variable -> bool) :
  ∀ (names : list name) (s s' : Join_env.t),
    (∀ x : variable, test x = true -> knows (Join_env.source_env s) x) ->
    List.fold_left
      (fun env name =>
         let* env := env in
         match name with
         | Name_symbol _ => Some env
         | Name_var var =>
           if test var
           then
             Join_env.import_or_delay_n_way_join_type p fuel env var
               (Join_env.types_of_name_in_levels envs name)
           else Some env
         end)
      names (Some s)
    = Some s' ->
    keeps_scope s s'.
Proof.
  induction names as [ | n names IH]; intros s s' Htest Hfold.
  - injection Hfold as <-. apply keeps_scope_refl.
  - cbn [List.fold_left] in Hfold.
    assert (Hstep :
              ∀ s1 : Join_env.t,
                keeps_scope s s1 ->
                List.fold_left
                  (fun env name =>
                     let* env := env in
                     match name with
                     | Name_symbol _ => Some env
                     | Name_var var =>
                       if test var
                       then
                         Join_env.import_or_delay_n_way_join_type p fuel env
                           var (Join_env.types_of_name_in_levels envs name)
                       else Some env
                     end)
                  names (Some s1)
                = Some s' ->
                keeps_scope s s').
    { intros s1 Hkeeps1 Hrest.
      apply (keeps_scope_trans _ _ _ Hkeeps1).
      apply (IH s1 s'); [ | exact Hrest ].
      intros x Hx. rewrite (proj1 (proj2 (proj1 Hkeeps1))).
      exact (Htest x Hx). }
    destruct n as [var | sym].
    + cbn in Hfold. destruct (test var) eqn:Hvar.
      * destruct (Join_env.import_or_delay_n_way_join_type p fuel s var
                    (Join_env.types_of_name_in_levels envs (Name_var var)))
          as [s1 | ] eqn:Himport.
        -- exact (Hstep s1
                    (import_or_delay_keeps_scope p fuel s s1 var _
                       (or_introl (Htest var Hvar)) Himport)
                    Hfold).
        -- exfalso. clear -Hfold.
           induction names as [ | e names IH];
             [ discriminate | exact (IH Hfold) ].
      * exact (Hstep s (keeps_scope_refl s) Hfold).
    + exact (Hstep s (keeps_scope_refl s) Hfold).
Qed.

Lemma keeps_scope_scoped (s s' : Join_env.t) :
  keeps_scope s s' -> imports_listed s -> state_scoped s -> state_scoped s'.
Proof.
  intros [_ Hkeeps] Hl Hscoped. apply scoped_but_nothing.
  apply (Hkeeps Hl). apply scoped_but_nothing. exact Hscoped.
Qed.

Lemma valid_imports_listed (s : Join_env.t) :
  state_valid s -> imports_listed s.
Proof. intros [_ [_ [_ [Hlisted _]]]]. exact Hlisted. Qed.

Section Rounds_scoped.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

Hypothesis jt_keeps : join_of_types_keeps_scoped jt.

Lemma round_fold_scoped :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    state_scoped s ->
    List.fold_left (round_step jt choices) bindings (Some (relations, s))
    = Some (relations', s') ->
    state_scoped s'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hscoped Hfold.
  - cbn in Hfold. injection Hfold as _ <-. exact Hscoped.
  - cbn [List.fold_left] in Hfold.
    destruct (round_step jt choices (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    assert (Hvar_types : types_hold s var types).
    { exact (Htypes var types (or_introl eq_refl)). }
    destruct (round_fold_holds source jt choices jt_sound [(var, types)] s s1
                relations relations1
                (fun v t Hin =>
                   match Hin with
                   | or_introl Heq => Htypes v t (or_introl Heq)
                   | or_intror Hnone => match Hnone with end
                   end)
                Htrue Hinvariant Hkinded Hrelations Hstep)
      as [Hlater1 [Hholds1 Hrelations1]].
    assert (Hscoped1 : state_scoped s1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      destruct (join_call_holds source jt jt_sound s s0 var types joined Htrue
                  Hinvariant Hkinded Hvar_types Hjoin)
        as [Hlater0 [Hholds0 Hknown]].
      pose proof (jt_keeps s s0 _ joined Hjoin
                    (valid_imports_listed s (proj1 Htrue)) Hscoped)
        as Hscoped0.
      assert (Hvar0 : exists_in_target s0 var = true).
      { exact (later_exists _ _ _ Hlater0 (proj1 Hvar_types)). }
      assert (Hadd :
                ∀ (ty' : flambda_type) (s2 : Join_env.t),
                  (∀ y : variable,
                     y ∈ free_variables ty' -> exists_in_target s0 y = true) ->
                  Join_env.add_equation_in_target_env s0 var ty' = Some s2 ->
                  state_scoped s2).
      { intros ty' s2 Hty' Hadd. apply scoped_but_nothing.
        apply (proj2 (add_equation_scope _ _ _ _ Hadd)).
        - intros y Hy. left. exact (Hty' y Hy).
        - apply scoped_but_nothing. exact Hscoped0. }
      destruct joined as [ty | ].
      - destruct (Hknown ty eq_refl) as [Htarget _].
        destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ] eqn:Hrecover; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-.
          apply (Hadd ty' s2); [ | exact Hadd2 ].
          intros y Hy.
          destruct (recover_free_variables _ _ _ _ _ y Hrecover Hy)
            as [Hin | ->]; [ exact (Htarget y Hin) | exact Hvar0 ].
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd2; [ | discriminate ].
          injection Hstep as _ <-. exact (Hadd ty s2 Htarget Hadd2).
      - injection Hstep as _ <-. exact Hscoped0. }
    exact (IH s1 s' relations1 relations'
             (fun v t Hin =>
                later_types_hold _ _ _ _ Hlater1 (Htypes v t (or_intror Hin)))
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hscoped1
             Hfold).
Qed.

Lemma round_scoped (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
     lookup var equations = Some types -> types_hold s var types) ->
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  relations_hold s relations ->
  state_scoped s ->
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  state_scoped s'.
Proof.
  intros Htypes Htrue Hinvariant Hkinded Hrelations Hscoped Hround.
  apply (round_fold_scoped (compact equations) s s' relations relations');
    [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Hrelations
      | exact Hscoped | exact Hround ].
  intros var types Hin. apply Htypes. apply in_compact. exact Hin.
Qed.

Lemma delayed_equations_scoped :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    state_scoped s ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    state_scoped s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hscoped Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. exact Hscoped.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    destruct (round_holds source jt choices jt_sound s s1 equations relations
                relations1 Htypes Htrue Hinvariant Hkinded Hrelations Hround)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (round_scoped s s1 equations relations relations1 Htypes Htrue
                  Hinvariant Hkinded Hrelations Hscoped Hround) as Hscoped1.
    assert (Htypes1 :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var (Join_env.new_equations_in_joined_envs s1 s)
                = Some types ->
                types_hold s1 var types).
    { intros var types Hvar.
      unfold Join_env.new_equations_in_joined_envs in Hvar.
      rewrite lookup_filter_bindings in Hvar.
      destruct (lookup var (Join_env.types_in_joined_envs s1))
        as [types1 | ] eqn:Hlookup; [ | discriminate ].
      destruct (negb (is_bound var (Join_env.types_in_joined_envs s)));
        [ | discriminate ].
      injection Hvar as <-.
      exact (proj1 (proj2 Hholds1) var types1 Hlookup). }
    exact (IH s1 s' _ relations1 relations' Htypes1
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hscoped1
             Hloop).
Qed.

Lemma loop_scoped (p : Join_env.parameters) :
  expand_const_sound p ->
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    state_scoped s ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    state_scoped s'.
Proof.
  intros Hconst.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Hrelations Hscoped Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. exact Hscoped.
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as _ <-. exact Hscoped. }
    destruct (Join_env.equations_of_existentials p s (compact existentials))
      as [equations | ] eqn:Hequations; [ | discriminate ].
    destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                relations equations)
      as [[relations1 s1] | ] eqn:Hdelayed; [ | discriminate ].
    assert (Htypes :
              ∀ (var : variable)
                (types : Keyed.finite_map Index.t flambda_type),
                lookup var equations = Some types -> types_hold s var types).
    { intros var types Hvar.
      destruct (equations_of_existentials_lookup p s _ _ Hequations _ _ Hvar)
        as [simples [kind [Hin Hfind]]].
      apply in_compact in Hin.
      exact (existential_types_hold p s var simples kind types Hconst
               (proj1 Htrue) (proj1 (proj2 Htrue)) Hinvariant Hkinded
               (Hdefinitions _ _ Hin) Hfind). }
    destruct (delayed_equations_hold source jt choices jt_sound fuel s s1
                equations relations relations1 Htypes Htrue Hinvariant Hkinded
                Hrelations Hdelayed)
      as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (delayed_equations_scoped fuel s s1 equations relations
                  relations1 Htypes Htrue Hinvariant Hkinded Hrelations
                  Hscoped Hdelayed) as Hscoped1.
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
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hscoped1
             Hloop).
Qed.

Lemma first_phases_scope (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  expand_const_sound p ->
  (∀ x : variable, knows source x -> x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     knows use x ->
     x ∈ avoid) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_invariant use) ->
  (∀ (index : Index.t) (use : Typing_env.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     typing_env_kinded use) ->
  levels_found envs ->
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
     (kind : flambda_kind) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Joined_envs.get_nth_joined_env envs index = Some use ->
     simple_of_kind use (Var x) kind ->
     Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
     = Some (ty, binding_time_and_mode) ->
     FlambdaTypeGrammar.kind ty = kind) ->
  Join_env.join_aliases_into_bindings p fuel
    (Join_env.create envs
       (Bindings_in_target_env.from_source_env avoid source))
    envs
  = Some s1 ->
  Join_env.n_way_join_delayed_equations fuel choices jt s1 []
    (Join_env.new_equations_in_joined_envs s1
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations2, s2) ->
  Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
    relations2
    (Join_env.new_definitions_of_existentials s2
       (Join_env.create envs
          (Bindings_in_target_env.from_source_env avoid source)))
  = Some (relations3, s3) ->
  state_scoped s3.
Proof.
  intros Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels Hkinds
    Haliases Hdelayed Hloop.
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
  destruct (join_aliases_true p fuel source s0 s1 Hfresh Heqb Htrue0 Hstate0
              Hinvariant Hkinded Hlevels Hknown0 Haliases)
    as [Htrue1 Hgrows1].
  assert (Hlater1 : later s0 s1).
  { split; [ exact (proj1 Htrue0) | ]. split; [ exact (proj1 Htrue1) | ].
    split; [ exact Hgrows1 | ].
    rewrite (proj2 (proj2 Htrue1)). symmetry. exact (proj2 (proj2 Htrue0)). }
  assert (Hinvariant0 : envs_invariant s0) by exact Hinvariant.
  assert (Hkinded0 : envs_kinded s0) by exact Hkinded.
  assert (Hscoped0 : state_scoped s0).
  { split.
    - intros x ty Hx. discriminate.
    - intros e x Hx. destruct Hx. }
  assert (Hscoped1 : state_scoped s1).
  { apply (keeps_scope_scoped s0 s1);
      [ | exact (valid_imports_listed s0 (proj1 Htrue0)) | exact Hscoped0 ].
    unfold Join_env.join_aliases_into_bindings in Haliases.
    refine (join_aliases_fold_keeps_scope p fuel envs _ _ s0 s1 _ Haliases).
    intros x Hx. rewrite <- kind_in_target_env_exists in Hx.
    rewrite (proj2 (proj2 Htrue0)). exact (Hknown0 x Hx). }
  assert (Htypes1 :
            ∀ (var : variable)
              (types : Keyed.finite_map Index.t flambda_type),
              lookup var (Join_env.new_equations_in_joined_envs s1 s0)
              = Some types ->
              types_hold s1 var types).
  { intros var types Hvar.
    unfold Join_env.new_equations_in_joined_envs in Hvar.
    rewrite lookup_filter_bindings in Hvar.
    destruct (lookup var (Join_env.types_in_joined_envs s1))
      as [types1 | ] eqn:Hlookup; [ | discriminate ].
    destruct (negb (is_bound var (Join_env.types_in_joined_envs s0)));
      [ | discriminate ].
    injection Hvar as <-.
    exact (proj1 (proj2 (proj1 (proj2 Htrue1))) var types1 Hlookup). }
  assert (Hrelations1 : relations_hold s1 []).
  { intros r relation n [recorded [Hlookup _]]. discriminate. }
  destruct (delayed_equations_hold source jt choices jt_sound fuel s1 s2 _ []
              relations2 Htypes1 Htrue1
              (later_envs_invariant _ _ Hlater1 Hinvariant0)
              (later_envs_kinded _ _ Hlater1 Hkinded0) Hrelations1 Hdelayed)
    as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (delayed_equations_scoped fuel s1 s2 _ [] relations2 Htypes1
                Htrue1 (later_envs_invariant _ _ Hlater1 Hinvariant0)
                (later_envs_kinded _ _ Hlater1 Hkinded0) Hrelations1 Hscoped1
                Hdelayed) as Hscoped2.
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
  exact (loop_scoped p Hconst (Join_env.join_depth p) fuel s2 s3 relations2
           relations3 _ Hdefinitions2
           (later_true source s1 s2 Hlater2 Hholds2 (proj2 (proj2 Htrue1)))
           (later_envs_invariant _ _ Hlater02 Hinvariant0)
           (later_envs_kinded _ _ Hlater02 Hkinded0) Hrelations2 Hscoped2
           Hloop).
Qed.

End Rounds_scoped.

(* ---------------------------------------------------------------------- *)
(* The statements                                                          *)
(* ---------------------------------------------------------------------- *)

Theorem first_phases_scoped : first_phases_scoped_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hkeeps Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels
    Hkinds Haliases Hdelayed Hloop.
  exact (first_phases_scope source jt choices Hjt Hkeeps p fuel avoid envs s1
           s2 s3 relations2 relations3 Hfresh Heqb Hconst Hsource Huses
           Hinvariant Hkinded Hlevels Hkinds Haliases Hdelayed Hloop).
Qed.

Lemma keeps_scope_relation (s s' : Join_env.t) :
  keeps_scope s s' ->
  imports_listed s ->
  imports_listed s' ∧ (state_scoped s -> state_scoped s').
Proof.
  intros Hkeeps Hl. split; [ exact (proj1 (proj1 Hkeeps) Hl) | ].
  exact (keeps_scope_scoped s s' Hkeeps Hl).
Qed.

Theorem n_way_join_keeps_scoped : n_way_join_keeps_scoped_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb s s' ts joined Hstep Hl Hscoped.
  refine
    (proj2
       (n_way_join_keeps Join_env.t
          (fun s0 s0' =>
             imports_listed s0 ->
             imports_listed s0' ∧ (state_scoped s0 -> state_scoped s0'))
          Join_env.joined_env (Join_env.n_way_join_simples p fuel_of_env)
          (Join_env.import_type p fuel_of_env)
          (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
          reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
          _ _ _ _ _ s s' ts joined Hstep Hl)
       Hscoped).
  - intros s0 H0. split; [ exact H0 | intros H; exact H ].
  - intros s1 s2 s3 H12 H23 H1. destruct (H12 H1) as [H2 K12].
    destruct (H23 H2) as [H3 K23].
    split; [ exact H3 | intros H; exact (K23 (K12 H)) ].
  - intros s0 s0' kind simples result Hsimples.
    exact (keeps_scope_relation _ _
             (n_way_join_simples_keeps_scope p fuel_of_env s0 s0' kind simples
                result Hsimples)).
  - intros s0 s0' ty Himport.
    exact (keeps_scope_relation _ _
             (proj1 (import_type_keeps_scope p fuel_of_env s0 s0' ty Himport))).
  - intros s0 s0' extensions joined0 Hextension.
    exact (keeps_scope_relation _ _
             (n_way_join_env_extension_keeps_scope p fuel_of_env s0 s0'
                extensions joined0 Hextension)).
Qed.

Theorem scoped_aliases_not_made : scoped_aliases_not_made_statement.
Proof.
  intros s [_ [Hnew [_ [_ Himports]]]] Hfrom e x Hx Hexistential.
  destruct (Hnew x Hexistential) as [Hsource Huses].
  destruct (Hfrom e x Hx) as [H | H]; [ exact (Hsource H) | ].
  destruct (Himports x H) as [_ [index [use [Huse Hknows]]]].
  exact (Huses index use Huse Hknows).
Qed.

Theorem cut_and_n_way_join0_in_scope : cut_and_n_way_join0_in_scope_statement.
Proof.
  intros meet_env meet_expanded_head ops p fuel choices jt mh avoid menv
    target source envs definitions names Hjt Hkeeps Hfresh Heqb Hconst
    Hsource Huses Hinvariant Hkinded Hlevels Hkinds Hempty Hjoin.
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
  destruct (first_phases_true source jt choices p fuel avoid envs s1 s2 s3
              relations2 relations3 Hjt Hfresh Heqb Hconst Hsource Huses
              Hinvariant Hkinded Hlevels Hkinds Haliases Hdelayed Hloop)
    as [Htrue3 [Hrelations3 Hgrows3]].
  destruct (first_phases_scoped source jt choices p fuel avoid envs s1 s2 s3
              relations2 relations3 Hjt Hkeeps Hfresh Heqb Hconst Hsource
              Huses Hinvariant Hkinded Hlevels Hkinds Haliases Hdelayed Hloop)
    as [Hequations3 Hfrom3].
  assert (Hnonempty1 : relations_nonempty []).
  { intros r names' Hr. discriminate Hr. }
  pose proof (loop_relations_nonempty _ _ _ _ _ _ _ _ _ _
                (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                   Hnonempty1 Hdelayed)
                Hloop) as Hnonempty3.
  destruct (define_variables_in_scope meet_env meet_expanded_head ops fuel mh
              source s3 _ target' relations3 definitions' Htrue3
              Hrelations3 Hnonempty3 Hequations3
              (scoped_aliases_not_made s3 (proj1 Htrue3) Hfrom3) Hdefine)
    as [kept [equations [relation_equations [menv' [Hkept [Hadd1 [Hadd2
          [Hscope1 Hscope2]]]]]]]].
  exists s3, kept, equations, relation_equations, menv'.
  split; [ exact Htrue3 | ]. split; [ reflexivity | ].
  split; [ exact Hkept | ]. split; [ exact Hadd1 | ].
  split; [ exact Hadd2 | ]. split; [ exact Hscope1 | exact Hscope2 ].
Qed.
