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


(* Proofs of the statements of FlambdaJoinDefineSpec.v. Nothing in this
   file needs to be reviewed. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs
  FlambdaRangeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeProject FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinEnvProofs FlambdaJoinStateSpec
  FlambdaJoinStateProofs FlambdaTypeEqualSpec FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesKeepsSpec FlambdaJoinTypesKeepsProofs
  FlambdaJoinTypesWithEnvSpec FlambdaJoinTypesWithEnvProofs
  FlambdaJoinDefineSpec.
Import ListNotations.

(* ---------------------------------------------------------------------- *)
(* Traversals                                                              *)
(* ---------------------------------------------------------------------- *)

Lemma traverse_in {A B : Type} (f : A -> option B) :
  ∀ (l : list A) (l' : list B),
    traverse f l = Some l' ->
    ∀ y : B, In y l' -> ∃ x : A, In x l ∧ f x = Some y.
Proof.
  induction l as [ | x l IH]; intros l' H y Hy; cbn [traverse] in H.
  - injection H as <-. destruct Hy.
  - destruct (f x) as [y0 | ] eqn:Hx; [ | discriminate ].
    destruct (traverse f l) as [l0 | ]; [ | discriminate ].
    injection H as <-. destruct Hy as [<- | Hy].
    + exists x. split; [ left; reflexivity | exact Hx ].
    + destruct (IH _ eq_refl y Hy) as [x' [Hin Hf]].
      exists x'. split; [ right; exact Hin | exact Hf ].
Qed.

(* A traversal of the bindings of a map that keeps the keys: what is
   found after it is what is made of what was found before. *)
Lemma traverse_lookup {K A B : Type} (eqb : K -> K -> bool)
    (f : K * A -> option (K * B)) :
  (∀ (b : K * A) (b' : K * B), f b = Some b' -> fst b' = fst b) ->
  ∀ (l : list (K * A)) (l' : list (K * B)),
    traverse f l = Some l' ->
    (∀ (key : K) (b' : B),
       Keyed.lookup eqb key l' = Some b' ->
       ∃ (key0 : K) (a : A),
         In (key0, a) l
         ∧ Keyed.lookup eqb key l = Some a
         ∧ f (key0, a) = Some (key0, b'))
    ∧ ∀ (key : K) (a : A),
        Keyed.lookup eqb key l = Some a ->
        ∃ (key0 : K) (b' : B),
          In (key0, a) l
          ∧ Keyed.lookup eqb key l' = Some b'
          ∧ f (key0, a) = Some (key0, b').
Proof.
  intros Hkeys.
  induction l as [ | [k0 a0] l IH]; intros l' H; cbn [traverse] in H.
  - injection H as <-. split; intros key x Hx; discriminate.
  - destruct (f (k0, a0)) as [[k0' b0] | ] eqn:Hf; [ | discriminate ].
    destruct (traverse f l) as [l0 | ]; [ | discriminate ].
    injection H as <-.
    pose proof (Hkeys _ _ Hf) as Hk. cbn [fst] in Hk. subst k0'.
    destruct (IH _ eq_refl) as [Hback Hforth]. split.
    + intros key b' Hlookup. rewrite keyed_lookup_cons in Hlookup.
      rewrite keyed_lookup_cons. destruct (eqb key k0).
      * injection Hlookup as <-. exists k0, a0.
        split; [ left; reflexivity | split; [ reflexivity | exact Hf ] ].
      * destruct (Hback key b' Hlookup) as [key0 [a [Hin [Hl Hfa]]]].
        exists key0, a. split; [ right; exact Hin | split; assumption ].
    + intros key a Hlookup. rewrite keyed_lookup_cons in Hlookup.
      rewrite keyed_lookup_cons. destruct (eqb key k0).
      * injection Hlookup as <-. exists k0, b0.
        split; [ left; reflexivity | split; [ reflexivity | exact Hf ] ].
      * destruct (Hforth key a Hlookup) as [key0 [b' [Hin [Hl Hfa]]]].
        exists key0, b'. split; [ right; exact Hin | split; assumption ].
Qed.

Lemma traverse_fields (P : value -> flambda_type -> Prop)
    (f : flambda_type -> option flambda_type) :
  ∀ (tys tys' : list flambda_type) (fields : list value),
    traverse f tys = Some tys' ->
    (∀ (ty ty' : flambda_type) (v : value),
       In ty tys -> f ty = Some ty' -> P v ty -> P v ty') ->
    fields_satisfied P tys fields ->
    fields_satisfied P tys' fields.
Proof.
  induction tys as [ | ty tys IH]; intros tys' fields H Himpl Hsatisfied;
    cbn [traverse] in H.
  - injection H as <-. simpl. auto.
  - destruct (f ty) as [ty' | ] eqn:Hty; [ | discriminate ].
    destruct (traverse f tys) as [tys0 | ] eqn:Htys; [ | discriminate ].
    injection H as <-. destruct fields as [ | field fields]; simpl; [ auto | ].
    simpl in Hsatisfied. destruct Hsatisfied as [Hfield Hfields]. split.
    + exact (Himpl ty ty' field (or_introl eq_refl) Hty Hfield).
    + apply (IH tys0 fields eq_refl); [ | exact Hfields ].
      intros t t' v Hin. exact (Himpl t t' v (or_intror Hin)).
Qed.

Section Project_sound.

Context (statics : static_env) (vars : env) (to_project : list variable)
  (expand : variable -> option flambda_type).

(* The value that a variable to remove has, if it has one, has the
   type put for the variable. *)
Hypothesis expand_true :
  ∀ (e : variable) (ty : flambda_type) (w : value),
    among e to_project = true ->
    expand e = Some ty ->
    lookup e vars = Some w ->
    type_satisfied statics vars w ty.

Lemma project_relation_sound (x x' : option variable) (n : Z) :
  project_relation to_project expand x = Some x' ->
  variable_is vars x n ->
  variable_is vars x' n.
Proof.
  unfold project_relation. destruct x as [e | ].
  2:{ intros H. injection H as <-. exact (fun H => H). }
  destruct (among e to_project) eqn:Hproject; cbn [negb].
  2:{ intros H. injection H as <-. exact (fun H => H). }
  destruct (expand e) as [ty | ] eqn:Hexpand; [ | discriminate ].
  intros H Hn. injection H as <-.
  pose proof (expand_true e ty _ Hproject Hexpand Hn) as Hty.
  destruct (get_alias_opt ty) as [[y | sym | c] | ] eqn:Halias; try exact I.
  exact (proj1 (proj1 (get_alias_satisfied statics vars _ ty _ Halias) Hty)).
Qed.

Lemma project_inverse_relation_sound (pair pair' : Relation.t * name)
    (l : list (Relation.t * name)) (i : Z) :
  project_inverse_relation to_project expand pair = Some l ->
  In pair' l ->
  (∃ w : value,
     value_of_name vars (snd pair) = Some w
     ∧ relation_holds statics (fst pair) w i) ->
  ∃ w : value,
    value_of_name vars (snd pair') = Some w
    ∧ relation_holds statics (fst pair') w i.
Proof.
  destruct pair as [rel n]. unfold project_inverse_relation. cbn [fst snd].
  destruct n as [e | sym].
  2:{ intros H Hin Hold. injection H as <-. destruct Hin as [<- | []].
      exact Hold. }
  destruct (among e to_project) eqn:Hproject; cbn [negb].
  2:{ intros H Hin Hold. injection H as <-. destruct Hin as [<- | []].
      exact Hold. }
  destruct (expand e) as [ty | ] eqn:Hexpand; [ | discriminate ].
  intros H Hin [w [Hw Hrel]]. injection H as <-.
  pose proof (expand_true e ty w Hproject Hexpand Hw) as Hty.
  destruct (get_alias_opt ty) as [[y | sym | c] | ] eqn:Halias;
    try (destruct Hin; fail).
  - destruct Hin as [<- | []]. exists w. split; [ | exact Hrel ].
    exact (proj1 (proj1 (get_alias_satisfied statics vars w ty _ Halias) Hty)).
  - destruct Hin as [<- | []]. exists w. split; [ | exact Hrel ].
    exact (proj1 (proj1 (get_alias_satisfied statics vars w ty _ Halias) Hty)).
Qed.

Lemma project_naked_immediate_sound (head head' : head_of_kind_naked_immediate)
    (v : value) :
  project_head_of_kind_naked_immediate to_project expand head = Some head' ->
  head_of_kind_naked_immediate_satisfied statics vars v head ->
  head_of_kind_naked_immediate_satisfied statics vars v head'.
Proof.
  unfold project_head_of_kind_naked_immediate.
  destruct (traverse (project_inverse_relation to_project expand)
              (inverse_relations head)) as [relations | ] eqn:Htraverse;
    [ | discriminate ].
  intros H [i [Hv [Hnumbers Hrelations]]]. injection H as <-.
  exists i. cbn [naked_immediates inverse_relations].
  split; [ exact Hv | ]. split; [ exact Hnumbers | ].
  intros rel n Hin. apply in_concat in Hin. destruct Hin as [l [Hl Hin]].
  destruct (traverse_in _ _ _ Htraverse l Hl) as [[rel0 n0] [Hpair Hf]].
  exact (project_inverse_relation_sound (rel0, n0) (rel, n) l i Hf Hin
           (Hrelations rel0 n0 Hpair)).
Qed.

Lemma project_descr_sound {head : Type} (k : flambda_kind)
    (project_head : head -> option head)
    (of_expanded : flambda_type -> option (type_descr head))
    (wrap : type_descr head -> flambda_type)
    (head_satisfied : head -> Prop) (v : value) (ty ty' : type_descr head) :
  (∀ (expanded : flambda_type) (d : type_descr head),
     of_expanded expanded = Some d -> expanded = wrap d) ->
  (∀ d : type_descr head,
     type_satisfied statics vars v (wrap d) ->
     descr_satisfied vars k head_satisfied v d) ->
  (∀ h h' : head,
     ty = Ok (No_alias h) ->
     project_head h = Some h' ->
     head_satisfied h ->
     head_satisfied h') ->
  project_descr to_project expand project_head of_expanded ty = Some ty' ->
  descr_satisfied vars k head_satisfied v ty ->
  descr_satisfied vars k head_satisfied v ty'.
Proof.
  intros Hof Hwrap Hhead. unfold project_descr.
  destruct ty as [ | [h | [e | sym | c]] | ].
  - intros H. injection H as <-. exact (fun H => H).
  - destruct (project_head h) as [h' | ] eqn:Hh; [ | discriminate ].
    intros H. injection H as <-. simpl. exact (Hhead h h' eq_refl Hh).
  - destruct (among e to_project) eqn:Hproject.
    + destruct (expand e) as [expanded | ] eqn:Hexpand; [ | discriminate ].
      intros Hd [Heval Hkind]. apply Hwrap. rewrite <- (Hof _ _ Hd).
      exact (expand_true e expanded v Hproject Hexpand Heval).
    + intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
  - intros H. injection H as <-. exact (fun H => H).
Qed.

Definition projected (n : name) : bool :=
  match n with
  | Name_var e => among e to_project
  | Name_symbol _ => false
  end.

(* The equations of an extension after the projection: one that is
   found is the projection of the one found before, on a name that is
   not removed. *)
Lemma project_equations_lookup
    (f : name * flambda_type -> option (list (name * flambda_type)))
    (project : flambda_type -> option flambda_type) :
  (∀ (n : name) (ty : flambda_type),
     f (n, ty)
     = if projected n
       then Some []
       else
         match project ty with
         | Some ty' => Some [(n, ty')]
         | None => None
         end) ->
  ∀ (equations : list (name * flambda_type))
    (lists : list (list (name * flambda_type))),
    traverse f equations = Some lists ->
    ∀ (n : name) (ty' : flambda_type),
      Keyed.lookup name_eqb n (concat lists) = Some ty' ->
      projected n = false
      ∧ ∃ ty : flambda_type,
          In (n, ty) equations
          ∧ Keyed.lookup name_eqb n equations = Some ty
          ∧ project ty = Some ty'.
Proof.
  intros Hf.
  induction equations as [ | [n0 ty0] rest IH]; intros lists H n ty' Hlookup;
    cbn [traverse] in H.
  - injection H as <-. discriminate.
  - rewrite Hf in H. destruct (projected n0) eqn:Hp0.
    + destruct (traverse f rest) as [lists0 | ] eqn:Hrest; [ | discriminate ].
      injection H as <-. cbn [concat app] in Hlookup.
      destruct (IH _ eq_refl n ty' Hlookup) as [Hp [ty [Hin [Hl Hproj]]]].
      split; [ exact Hp | ]. exists ty. split; [ right; exact Hin | ].
      split; [ | exact Hproj ].
      rewrite keyed_lookup_cons. destruct (name_eqb n n0) eqn:Heq;
        [ | exact Hl ].
      apply name_eqb_decides_equality in Heq. subst n0. congruence.
    + destruct (project ty0) as [ty0' | ] eqn:Hty0; [ | discriminate ].
      destruct (traverse f rest) as [lists0 | ] eqn:Hrest; [ | discriminate ].
      injection H as <-. cbn [concat app] in Hlookup.
      rewrite keyed_lookup_cons in Hlookup.
      destruct (name_eqb n n0) eqn:Heq.
      * injection Hlookup as <-. apply name_eqb_decides_equality in Heq.
        subst n0. split; [ exact Hp0 | ]. exists ty0.
        split; [ left; reflexivity | ]. split; [ | exact Hty0 ].
        rewrite keyed_lookup_cons.
        rewrite (proj2 (name_eqb_decides_equality n n) eq_refl). reflexivity.
      * destruct (IH _ eq_refl n ty' Hlookup) as [Hp [ty [Hin [Hl Hproj]]]].
        split; [ exact Hp | ]. exists ty. split; [ right; exact Hin | ].
        split; [ | exact Hproj ]. rewrite keyed_lookup_cons, Heq. exact Hl.
Qed.

Lemma of_expanded_naked_number (k : naked_number_kind)
    (expanded : flambda_type) (d : type_descr (list Z)) :
  naked_number_of_expanded k expanded = Some d ->
  expanded = Type_naked_number k d.
Proof.
  destruct expanded as [d' | d' | k' d' | d' | d']; cbn; try discriminate.
  destruct (kind_eqb (Naked_number k) (Naked_number k')) eqn:Hkind;
    [ | discriminate ].
  intros H. injection H as <-. apply kind_eqb_eq in Hkind.
  injection Hkind as <-. reflexivity.
Qed.

Lemma project_all_sound :
  (∀ (t t' : flambda_type) (v : value),
     project_variables_out to_project expand t = Some t' ->
     type_satisfied statics vars v t ->
     type_satisfied statics vars v t')
  ∧ (∀ (head head' : head_of_kind_value) (v : value),
       project_head_of_kind_value to_project expand head = Some head' ->
       head_of_kind_value_satisfied statics vars v head ->
       head_of_kind_value_satisfied statics vars v head')
  ∧ (∀ (head head' : head_of_kind_value_non_null) (v : value),
       project_head_of_kind_value_non_null to_project expand head
       = Some head' ->
       head_of_kind_value_non_null_satisfied statics vars v head ->
       head_of_kind_value_non_null_satisfied statics vars v head')
  ∧ (∀ (blocks blocks' : row_like_for_blocks) (tag : Z)
       (fields : list value),
       project_row_like_for_blocks to_project expand blocks = Some blocks' ->
       row_like_for_blocks_satisfied statics vars tag fields blocks ->
       row_like_for_blocks_satisfied statics vars tag fields blocks')
  ∧ (∀ (closures closures' : row_like_for_closures) (v : value)
       (slot : function_slot) (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
       project_row_like_for_closures to_project expand closures
       = Some closures' ->
       row_like_for_closures_satisfied statics vars v slot functions
         value_slots closures ->
       row_like_for_closures_satisfied statics vars v slot functions
         value_slots closures')
  ∧ (∀ (entry entry' : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
       project_closures_entry to_project expand entry = Some entry' ->
       closures_entry_satisfied statics vars v slot functions value_slots
         entry ->
       closures_entry_satisfied statics vars v slot functions value_slots
         entry')
  ∧ (∀ extension extension' : env_extension,
       project_env_extension to_project expand extension = Some extension' ->
       env_extension_satisfied statics vars extension ->
       env_extension_satisfied statics vars extension').
Proof.
  apply (types_induction
           (fun t =>
              ∀ (t' : flambda_type) (v : value),
                project_variables_out to_project expand t = Some t' ->
                type_satisfied statics vars v t ->
                type_satisfied statics vars v t')
           (fun head =>
              ∀ (head' : head_of_kind_value) (v : value),
                project_head_of_kind_value to_project expand head
                = Some head' ->
                head_of_kind_value_satisfied statics vars v head ->
                head_of_kind_value_satisfied statics vars v head')
           (fun head =>
              ∀ (head' : head_of_kind_value_non_null) (v : value),
                project_head_of_kind_value_non_null to_project expand head
                = Some head' ->
                head_of_kind_value_non_null_satisfied statics vars v head ->
                head_of_kind_value_non_null_satisfied statics vars v head')
           (fun blocks =>
              ∀ (blocks' : row_like_for_blocks) (tag : Z)
                (fields : list value),
                project_row_like_for_blocks to_project expand blocks
                = Some blocks' ->
                row_like_for_blocks_satisfied statics vars tag fields blocks ->
                row_like_for_blocks_satisfied statics vars tag fields blocks')
           (fun closures =>
              ∀ (closures' : row_like_for_closures) (v : value)
                (slot : function_slot)
                (functions : list (function_slot * code_id))
                (value_slots : list (value_slot * value)),
                project_row_like_for_closures to_project expand closures
                = Some closures' ->
                row_like_for_closures_satisfied statics vars v slot functions
                  value_slots closures ->
                row_like_for_closures_satisfied statics vars v slot functions
                  value_slots closures')
           (fun entry =>
              ∀ (entry' : closures_entry) (v : value) (slot : function_slot)
                (functions : list (function_slot * code_id))
                (value_slots : list (value_slot * value)),
                project_closures_entry to_project expand entry = Some entry'
                ->
                closures_entry_satisfied statics vars v slot functions
                  value_slots entry ->
                closures_entry_satisfied statics vars v slot functions
                  value_slots entry')
           (fun extension =>
              ∀ extension' : env_extension,
                project_env_extension to_project expand extension
                = Some extension' ->
                env_extension_satisfied statics vars extension ->
                env_extension_satisfied statics vars extension')).
  - (* Type_value *)
    intros ty Hhead t' v H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_value to_project expand)
                value_of_expanded ty) as [ty' | ] eqn:Hd; [ | discriminate ].
    injection H as <-. rewrite !type_satisfied_value.
    refine (project_descr_sound Value _ _ Type_value _ v ty ty' _ _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros d Hsat. rewrite type_satisfied_value in Hsat. exact Hsat.
    + intros h h' Heq Hh. subst ty. exact (Hhead h' v Hh).
  - (* Type_naked_immediate *)
    intros ty t' v H. simpl in H.
    destruct (project_descr to_project expand
                (project_head_of_kind_naked_immediate to_project expand)
                naked_immediate_of_expanded ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-. rewrite !type_satisfied_naked_immediate.
    refine (project_descr_sound (Naked_number Kind_naked_immediate) _ _
              Type_naked_immediate _ v ty ty' _ _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros d Hsat. rewrite type_satisfied_naked_immediate in Hsat.
      exact Hsat.
    + intros h h' _ Hh. exact (project_naked_immediate_sound h h' v Hh).
  - (* Type_naked_number *)
    intros k ty t' v H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                (naked_number_of_expanded k) ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-. rewrite !type_satisfied_naked_number.
    refine (project_descr_sound (Naked_number k) _ _ (Type_naked_number k) _
              v ty ty' _ _ _ Hd).
    + intros expanded d. exact (of_expanded_naked_number k expanded d).
    + intros d Hsat. rewrite type_satisfied_naked_number in Hsat. exact Hsat.
    + intros h h' _ Hh. injection Hh as <-. exact (fun H => H).
  - (* Type_rec_info *)
    intros ty t' v H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                rec_info_of_expanded ty) as [ty' | ] eqn:Hd;
      [ | discriminate ].
    injection H as <-. rewrite !type_satisfied_rec_info.
    refine (project_descr_sound Rec_info _ _ Type_rec_info _ v ty ty' _ _ _
              Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros d Hsat. rewrite type_satisfied_rec_info in Hsat. exact Hsat.
    + intros h h' _ Hh. injection Hh as <-. exact (fun H => H).
  - (* Type_region *)
    intros ty t' v H. simpl in H.
    destruct (project_descr to_project expand (fun head => Some head)
                region_of_expanded ty) as [ty' | ] eqn:Hd; [ | discriminate ].
    injection H as <-. rewrite !type_satisfied_region.
    refine (project_descr_sound Region _ _ Type_region _ v ty ty' _ _ _ Hd).
    + intros expanded d. destruct expanded; cbn; intros H; try discriminate.
      injection H as <-. reflexivity.
    + intros d Hsat. rewrite type_satisfied_region in Hsat. exact Hsat.
    + intros h h' _ Hh. injection Hh as <-. exact (fun H => H).
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null head' v H.
    simpl in H.
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
                (match is_null with
                 | Not_null => False
                 | Maybe_null x => variable_is vars x 1
                 end ->
                 match is_null' with
                 | Not_null => False
                 | Maybe_null x => variable_is vars x 1
                 end)
                ∧ (match is_null with
                   | Not_null => True
                   | Maybe_null x => variable_is vars x 0
                   end ->
                   match is_null' with
                   | Not_null => True
                   | Maybe_null x => variable_is vars x 0
                   end)).
    { intros is_null' Hn. destruct is_null as [ | x].
      - injection Hn as <-. split; exact (fun H => H).
      - destruct (project_relation to_project expand x) as [x' | ] eqn:Hx;
          [ | discriminate ].
        injection Hn as <-.
        split; exact (project_relation_sound x x' _ Hx). }
    destruct (match is_null with
              | Not_null => Some Not_null
              | Maybe_null is_null_var =>
                let* is_null_var :=
                  project_relation to_project expand is_null_var
                in
                Some (Maybe_null is_null_var)
              end) as [is_null' | ] eqn:His_null; [ | discriminate ].
    destruct (Hnull is_null' eq_refl) as [Hone Hzero].
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
                match non_null with
                | Unknown => kind_of_value v = Value
                | Bottom => False
                | Ok head =>
                  head_of_kind_value_non_null_satisfied statics vars v head
                end ->
                match non_null' with
                | Unknown => kind_of_value v = Value
                | Bottom => False
                | Ok head =>
                  head_of_kind_value_non_null_satisfied statics vars v head
                end).
    { intros non_null' Hn. destruct non_null as [ | head | ].
      - injection Hn as <-. exact (fun H => H).
      - destruct (project_head_of_kind_value_non_null to_project expand head)
          as [head0 | ] eqn:Hhead0; [ | discriminate ].
        injection Hn as <-. exact (Hnon_null head0 v eq_refl).
      - injection Hn as <-. exact (fun H => H). }
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
    injection H as <-. rewrite !head_of_kind_value_satisfied_eq.
    intros [[Hv Hn] | [Hv [Hh Hn]]].
    + left. split; [ exact Hv | exact (Hone Hn) ].
    + right. split; [ exact Hv | ].
      split; [ exact (Hhead non_null' eq_refl Hh) | exact (Hzero Hn) ].
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions head' v H.
    simpl in H.
    destruct (project_relation to_project expand is_int) as [is_int' | ]
      eqn:His_int; [ | discriminate ].
    destruct (project_relation to_project expand get_tag) as [get_tag' | ]
      eqn:Hget_tag; [ | discriminate ].
    assert (Himm :
              ∀ (immediates' : Or_unknown.t flambda_type) (w : value),
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
                | Or_unknown.Unknown => True
                | Or_unknown.Known ty => type_satisfied statics vars w ty
                end ->
                match immediates' with
                | Or_unknown.Unknown => True
                | Or_unknown.Known ty => type_satisfied statics vars w ty
                end).
    { intros immediates' w Hi. destruct immediates as [imm | ].
      - destruct (project_variables_out to_project expand imm) as [imm' | ]
          eqn:Himm'; [ | discriminate ].
        injection Hi as <-. exact (Himmediates imm' w eq_refl).
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
              ∀ (blocks' : Or_unknown.t row_like_for_blocks) (tag : Z)
                (fields : list value),
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
                | Or_unknown.Unknown => True
                | Or_unknown.Known blocks =>
                  row_like_for_blocks_satisfied statics vars tag fields blocks
                end ->
                match blocks' with
                | Or_unknown.Unknown => True
                | Or_unknown.Known blocks =>
                  row_like_for_blocks_satisfied statics vars tag fields blocks
                end).
    { intros blocks' tag fields Hb. destruct blocks as [row | ].
      - destruct (project_row_like_for_blocks to_project expand row)
          as [row' | ] eqn:Hrow'; [ | discriminate ].
        injection Hb as <-. exact (Hblocks row' tag fields eq_refl).
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
                (match extensions with
                 | No_extensions => True
                 | Ext when_immediate _ =>
                   env_extension_satisfied statics vars when_immediate
                 end ->
                 match extensions' with
                 | No_extensions => True
                 | Ext when_immediate _ =>
                   env_extension_satisfied statics vars when_immediate
                 end)
                ∧ (match extensions with
                   | No_extensions => True
                   | Ext _ when_block =>
                     env_extension_satisfied statics vars when_block
                   end ->
                   match extensions' with
                   | No_extensions => True
                   | Ext _ when_block =>
                     env_extension_satisfied statics vars when_block
                   end)).
    { intros extensions' He. destruct extensions as [ | wi wb].
      - injection He as <-. split; exact (fun H => H).
      - destruct (project_env_extension to_project expand wi) as [wi' | ]
          eqn:Hwi; [ | discriminate ].
        destruct (project_env_extension to_project expand wb) as [wb' | ]
          eqn:Hwb; [ | discriminate ].
        injection He as <-. destruct Hextensions as [IHwi IHwb].
        split; [ exact (IHwi wi' eq_refl) | exact (IHwb wb' eq_refl) ]. }
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
    destruct (Hext extensions' eq_refl) as [Hext_imm Hext_block].
    injection H as <-. rewrite !FlambdaTypeMeaningProofs.variant_satisfied.
    intros [[i [Hv [Hint [Himmediate Hextension]]]]
           | [tag [fields [Hblock [Hint [Htag [Hrows Hextension]]]]]]].
    + left. exists i. split; [ exact Hv | ].
      split; [ exact (project_relation_sound _ _ _ His_int Hint) | ].
      split; [ exact (Himm immediates' _ eq_refl Himmediate) | ].
      exact (Hext_imm Hextension).
    + right. exists tag, fields. split; [ exact Hblock | ].
      split; [ exact (project_relation_sound _ _ _ His_int Hint) | ].
      split; [ exact (project_relation_sound _ _ _ Hget_tag Htag) | ].
      split; [ exact (Hrow blocks' tag fields eq_refl Hrows) | ].
      exact (Hext_block Hextension).
  - (* Mutable_block *)
    intros alloc_mode head' v H. injection H as <-. exact (fun H => H).
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents head' v H.
    simpl in H.
    destruct (project_variables_out to_project expand contents)
      as [contents' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. rewrite !boxed_number_satisfied.
    intros [c [Hboxed [Hkind Hsat]]]. exists c.
    split; [ exact Hboxed | ]. split; [ exact Hkind | ].
    exact (Hcontents contents' _ eq_refl Hsat).
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures head' v H.
    simpl in H.
    destruct (project_row_like_for_closures to_project expand
                by_function_slot) as [closures' | ] eqn:Hc; [ | discriminate ].
    injection H as <-. rewrite !closures_satisfied.
    intros [slot [functions [value_slots [Hclosure Hrow]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    exact (Hclosures closures' v slot functions value_slots eq_refl Hrow).
  - (* String *)
    intros strs head' v H. injection H as <-. exact (fun H => H).
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother blocks' tag fields
      H.
    simpl in H.
    assert (Hcase :
              ∀ case case' : row_like_block_case,
                P_block_case
                  (fun t =>
                     ∀ (t' : flambda_type) (v : value),
                       project_variables_out to_project expand t = Some t' ->
                       type_satisfied statics vars v t ->
                       type_satisfied statics vars v t')
                  (fun extension =>
                     ∀ extension' : env_extension,
                       project_env_extension to_project expand extension
                       = Some extension' ->
                       env_extension_satisfied statics vars extension ->
                       env_extension_satisfied statics vars extension')
                  case ->
                project_block_case (project_variables_out to_project expand)
                  (project_env_extension to_project expand) case
                = Some case' ->
                row_like_block_case_satisfied statics vars tag fields case ->
                row_like_block_case_satisfied statics vars tag fields case').
    { intros case case' [Htypes Hextension] Hp [Hindex [Hfields Hext]].
      unfold project_block_case in Hp.
      destruct (project_env_extension to_project expand
                  (case_env_extension case)) as [ext' | ] eqn:Hext';
        [ | discriminate ].
      destruct (traverse (project_variables_out to_project expand)
                  (case_maps_to case)) as [maps' | ] eqn:Hmaps;
        [ | discriminate ].
      injection Hp as <-. split; [ exact Hindex | ]. split.
      - apply (traverse_fields _ _ _ _ _ Hmaps); [ | exact Hfields ].
        intros ty ty' w Hin Hty. exact (Htypes ty Hin ty' w Hty).
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
    destruct (traverse_lookup Z.eqb _ Hkeys _ _ Htraverse) as [Hback Hforth].
    assert (Hother' :
              ∃ other_tags' : Or_bottom.t row_like_block_case,
                Some (Row_like_for_blocks known_tags' other_tags' alloc_mode)
                = Some blocks'
                ∧ (match other_tags with
                   | Or_bottom.Ok case =>
                     row_like_block_case_satisfied statics vars tag fields
                       case
                   | Or_bottom.Bottom => False
                   end ->
                   match other_tags' with
                   | Or_bottom.Ok case =>
                     row_like_block_case_satisfied statics vars tag fields
                       case
                   | Or_bottom.Bottom => False
                   end)).
    { destruct other_tags as [case | ].
      - destruct (project_block_case _ _ case) as [case' | ] eqn:Hcase';
          [ | discriminate ].
        exists (Or_bottom.Ok case'). split; [ exact H | ].
        exact (Hcase case case' Hother Hcase').
      - exists Or_bottom.Bottom. split; [ exact H | exact (fun H => H) ]. }
    clear H. destruct Hother' as [other_tags' [H Hother']].
    injection H as <-. rewrite !row_like_for_blocks_find_iff.
    destruct (Keyed.lookup Z.eqb tag known_tags') as [[case' | ] | ]
      eqn:Hfind'.
    + destruct (Hback tag _ Hfind') as [key0 [a [Hin [Hfind Hf]]]].
      rewrite Hfind. cbn [fst snd] in Hf. destruct a as [case | ].
      * destruct (project_block_case _ _ case) as [case0 | ] eqn:Hcase0;
          [ | discriminate ].
        injection Hf as <-.
        exact (Hcase case case0 (Hknown key0 case Hin) Hcase0).
      * discriminate.
    + exact (fun _ => I).
    + destruct (Keyed.lookup Z.eqb tag known_tags) as [a | ] eqn:Hfind.
      * destruct (Hforth tag a Hfind) as [key0 [b' [_ [Hfind'' _]]]].
        congruence.
      * exact Hother'.
  - (* Row_like_for_closures *)
    intros known_closures Hknown closures' v slot functions value_slots H.
    simpl in H.
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
    destruct (traverse_lookup Pos.eqb _ Hkeys _ _ Htraverse) as [_ Hforth].
    injection H as <-. rewrite !row_like_for_closures_lookup_iff.
    intros [case [Hlookup [Hindex [Hentry Hextension]]]].
    destruct (Hforth slot case Hlookup) as [key0 [case' [Hin [Hlookup' Hf]]]].
    exists case'. split; [ exact Hlookup' | ].
    cbn [fst snd] in Hf.
    destruct (project_env_extension to_project expand
                (case_env_extension case)) as [ext' | ] eqn:Hext';
      [ | discriminate ].
    destruct (project_closures_entry to_project expand (case_maps_to case))
      as [maps' | ] eqn:Hmaps; [ | discriminate ].
    injection Hf as <-. destruct (Hknown key0 case Hin) as [IHentry IHext].
    split; [ exact Hindex | ]. split.
    + exact (IHentry maps' v slot functions value_slots Hmaps Hentry).
    + exact (IHext ext' Hext' Hextension).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      entry' v slot functions value_slots H.
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
    injection H as <-. rewrite !closures_entry_lookup_iff.
    intros [Hfunctions [Hclosure_types Hvalue_slots]].
    split; [ exact Hfunctions | ]. split.
    + intros g ty' closure Hlookup' Hproject.
      destruct (Hback_c g ty' Hlookup') as [key0 [ty [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf.
      destruct (project_variables_out to_project expand ty) as [ty0 | ]
        eqn:Hty; [ | discriminate ].
      injection Hf as <-.
      exact (Hclosures key0 ty Hin ty0 closure Hty
               (Hclosure_types g ty closure Hlookup Hproject)).
    + intros w ty' contents Hlookup' Hcontents.
      destruct (Hback_s w ty' Hlookup') as [key0 [ty [Hin [Hlookup Hf]]]].
      cbn [fst snd] in Hf.
      destruct (project_variables_out to_project expand ty) as [ty0 | ]
        eqn:Hty; [ | discriminate ].
      injection Hf as <-.
      exact (Hslots key0 ty Hin ty0 contents Hty
               (Hvalue_slots w ty contents Hlookup Hcontents)).
  - (* Env_extension *)
    intros equations Htypes extension' H.
    simpl in H.
    match type of H with
    | context [traverse ?f equations] =>
      destruct (traverse f equations) as [lists | ] eqn:Htraverse;
        [ | discriminate ];
      assert (Hf :
                ∀ (n : name) (ty : flambda_type),
                  f (n, ty)
                  = if projected n
                    then Some []
                    else
                      match project_variables_out to_project expand ty with
                      | Some ty' => Some [(n, ty')]
                      | None => None
                      end)
    end.
    { intros [e | sym] ty; cbn [fst snd projected];
        [ destruct (among e to_project) | ]; reflexivity. }
    injection H as <-. rewrite !env_extension_lookup_iff.
    intros Hold n ty' Hlookup'.
    destruct (project_equations_lookup _ _ Hf _ _ Htraverse n ty' Hlookup')
      as [_ [ty [Hin [Hlookup Hproject]]]].
    destruct (Hold n ty Hlookup) as [v [Hv Hty]]. exists v.
    split; [ exact Hv | exact (Htypes n ty Hin ty' v Hproject Hty) ].
Qed.

End Project_sound.

Theorem project_variables_out_sound : project_variables_out_sound_statement.
Proof.
  intros statics vars to_project expand t t' v Hexpand.
  exact (proj1 (project_all_sound statics vars to_project expand Hexpand) t
           t' v).
Qed.

Theorem project_env_extension_sound : project_env_extension_sound_statement.
Proof.
  intros statics vars to_project expand extension extension' Hexpand.
  destruct (project_all_sound statics vars to_project expand Hexpand)
    as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
  exact (Hextension extension extension').
Qed.

(* ====================================================================== *)
(* expand_variable                                                         *)
(* ====================================================================== *)

Lemma expand_variable_true (statics : static_env) (vars : env)
    (to_expand : finite_map flambda_type) :
  (∀ (e : variable) (ty : flambda_type) (w : value),
     lookup e to_expand = Some ty ->
     lookup e vars = Some w ->
     type_satisfied statics vars w ty) ->
  ∀ (fuel : nat) (e : variable) (ty : flambda_type) (w : value),
    Join_env.expand_variable fuel to_expand e = Some ty ->
    lookup e vars = Some w ->
    type_satisfied statics vars w ty.
Proof.
  intros Hbase. induction fuel as [ | fuel IH]; intros e ty w Hexpand Hw;
    cbn [Join_env.expand_variable] in Hexpand; [ discriminate | ].
  destruct (lookup e to_expand) as [ty0 | ] eqn:Hty0; [ | discriminate ].
  cbv beta iota in Hexpand.
  pose proof (Hbase e ty0 w Hty0 Hw) as Hsat.
  destruct (get_alias_opt ty0) as [[x | sym | c] | ] eqn:Halias.
  - destruct (among x (domain to_expand)).
    + apply (IH x ty w Hexpand).
      apply (get_alias_satisfied statics vars w ty0 (Var x) Halias) in Hsat.
      exact (proj1 Hsat).
    + injection Hexpand as <-. exact Hsat.
  - injection Hexpand as <-. exact Hsat.
  - injection Hexpand as <-. exact Hsat.
  - assert (Hexp :
              ∀ (e' : variable) (ty' : flambda_type) (w' : value),
                among e' (domain to_expand) = true ->
                Join_env.expand_variable fuel to_expand e' = Some ty' ->
                lookup e' vars = Some w' ->
                type_satisfied statics vars w' ty').
    { intros e' ty' w' _ He' Hw'. exact (IH e' ty' w' He' Hw'). }
    destruct (project_all_sound statics vars (domain to_expand)
                (Join_env.expand_variable fuel to_expand) Hexp)
      as [Hproject _].
    exact (Hproject ty0 ty w Hexpand Hsat).
Qed.

(* ====================================================================== *)
(* The kind of the value of a variable that the join has made              *)
(* ====================================================================== *)

Lemma simple_of_kind_value (statics : static_env) (valuation : env)
    (use : Typing_env.t) (simple : simple) (kind : flambda_kind) (v : value) :
  typing_env_kinded use ->
  typing_env_satisfied_strictly statics valuation use ->
  simple_of_kind use simple kind ->
  eval_simple valuation simple = Some v ->
  kind_of_value v = kind.
Proof.
  intros Hkinded Hsatisfied Hkind Hv. destruct simple as [x | sym | c].
  - destruct Hkind as [ty Hfind]. unfold Typing_env.find in Hfind.
    unfold Typing_env.find_with_binding_time_and_mode' in Hfind.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types use)) as [[ty0 btm] | ] eqn:Hlookup;
      [ | discriminate ].
    destruct (Typing_env.check_optional_kind_matches ty0 (Some kind))
      eqn:Hcheck; [ | discriminate ].
    cbn in Hcheck. apply kind_eqb_eq in Hcheck. subst kind.
    apply (type_satisfied_kind statics valuation v ty0).
    exact (found_satisfied statics valuation use (Name_var x) ty0 btm v
             Hsatisfied Hlookup Hv).
  - cbn in Hv. injection Hv as <-.
    rewrite (symbol_kinded use sym kind Hkinded Hkind). reflexivity.
  - cbn in Hv. injection Hv as <-. cbn in Hkind. rewrite <- Hkind.
    exact (type_satisfied_kind statics valuation (Value_const c)
             (type_for_const c) (type_for_const_satisfied statics valuation c)).
Qed.

Lemma existential_value_kind (s : Join_env.t) (e : variable)
    (kind : flambda_kind) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env) (v : value) :
  state_valid s ->
  state_kinded s ->
  joined_envs_are typing_env_kinded s ->
  lookup e (Bindings_in_target_env.existential_variables (Join_env.bindings s))
  = Some kind ->
  Join_env.joined_env s index = Some use ->
  typing_env_satisfied_strictly statics valuation use ->
  lookup e (target_valuation s index use valuation) = Some v ->
  kind_of_value v = kind.
Proof.
  intros Hvalid Hkinded Henvs He Huse Hsatisfied Hv.
  pose proof Hvalid as [_ [_ [[Hrecorded [_ Hdefined]] _]]].
  assert (Hexistential : is_existential s e).
  { unfold is_existential. apply is_bound_lookup. exists kind. exact He. }
  pose proof (Hdefined e Hexistential) as Hbound.
  apply is_bound_lookup in Hbound. destruct Hbound as [[simples kind'] Hdef].
  pose proof (Hrecorded _ _ _ Hdef) as He'. rewrite He in He'.
  injection He' as <-.
  destruct (existential_value s e simples kind index use valuation v Hvalid
              Hdef Huse Hv) as [simple [Hfind Hvalue]].
  destruct Hkinded as [_ [_ Hdefinitions]].
  exact (simple_of_kind_value statics valuation use simple kind v
           (Henvs _ _ Huse) Hsatisfied
           (Hdefinitions e simples kind index simple use Hdef Hfind Huse)
           Hvalue).
Qed.

(* ====================================================================== *)
(* The fold over the variables that the join has made                      *)
(* ====================================================================== *)

Definition define_acc (meet_env : Type) : Type :=
  meet_env * finite_map flambda_type * finite_map flambda_type
  * Join_env.inverse_relations
  * finite_map (Simples_in_joined_envs.t * flambda_kind).

Definition define_step {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
    (unique : variable -> bool) (acc : option (define_acc meet_env))
    (binding : variable * flambda_kind) : option (define_acc meet_env) :=
  let* '(target_env, to_expand, equations, relations, definitions) := acc in
  let var := fst binding in
  let var_kind := snd binding in
  let aliases_of_var :=
    Aliases_of_existentials.aliases_of_existential_var
      (Join_env.aliases_of_existentials s) var
  in
  let* canon := Join_env.choose_alias tenv_of_aliases aliases_of_var in
  match canon with
  | None =>
    if unique var
    then
      let ty :=
        match lookup var equations with
        | Some ty => ty
        | None => unknown var_kind
        end
      in
      Some
        (target_env, (var, ty) :: to_expand, unbind var equations, relations,
         unbind var definitions)
    else
      Some
        (Join_env.add_variable_definition ops target_env var var_kind
           Name_mode.in_types,
         to_expand, equations, relations, definitions)
  | Some canon_var =>
    let demoted_aliases :=
      filter (fun alias => negb (Pos.eqb alias canon_var)) aliases_of_var
    in
    let definitions := Join_env.move_definition definitions var canon_var in
    let* equations := Join_env.move_equation equations var canon_var in
    let relations := Join_env.move_inverse_relation relations var canon_var in
    let ty := alias_type_of var_kind (Var canon_var) in
    let equations :=
      List.fold_left (fun equations alias => (alias, ty) :: equations)
        demoted_aliases equations
    in
    Some (target_env, (var, ty) :: to_expand, equations, relations, definitions)
  end.

Lemma fold_left_none {A B : Type} (f : option A -> B -> option A) :
  (∀ b : B, f None b = None) ->
  ∀ l : list B, List.fold_left f l None = None.
Proof.
  intros Hnone. induction l as [ | b l IH]; [ reflexivity | ].
  cbn [List.fold_left]. rewrite Hnone. exact IH.
Qed.

Lemma fold_left_invariant {A B : Type} (f : option A -> B -> option A)
    (invariant : A -> Prop) :
  (∀ b : B, f None b = None) ->
  ∀ l : list B,
    (∀ (a a' : A) (b : B),
       b ∈ l -> invariant a -> f (Some a) b = Some a' -> invariant a') ->
    ∀ a a' : A,
      invariant a -> List.fold_left f l (Some a) = Some a' -> invariant a'.
Proof.
  intros Hnone. induction l as [ | b l IH]; intros Hstep a a' Ha Hfold.
  - injection Hfold as <-. exact Ha.
  - cbn [List.fold_left] in Hfold.
    destruct (f (Some a) b) as [a1 | ] eqn:Hb.
    + apply (IH (fun a0 a0' b0 Hin => Hstep a0 a0' b0 (or_intror Hin)) a1 a'
               (Hstep a a1 b (or_introl eq_refl) Ha Hb) Hfold).
    + rewrite (fold_left_none f Hnone) in Hfold. discriminate.
Qed.

Lemma earliest_bound_in (tenv : Typing_env.t) :
  ∀ (candidates : list variable) (best c : variable),
    Join_env.earliest_bound tenv best candidates = Some c ->
    c = best ∨ c ∈ candidates.
Proof.
  induction candidates as [ | var candidates IH]; intros best c H;
    cbn [Join_env.earliest_bound] in H.
  - injection H as <-. left. reflexivity.
  - destruct (Typing_env.binding_time_and_mode tenv (Name_var var))
      as [time | ]; [ | discriminate ].
    destruct (Typing_env.binding_time_and_mode tenv (Name_var best))
      as [best_time | ]; [ | discriminate ].
    cbv beta iota in H.
    destruct (IH _ _ H) as [Heq | Hin]; [ | right; right; exact Hin ].
    destruct (Binding_time.strictly_earlier _ _).
    + right. left. symmetry. exact Heq.
    + left. exact Heq.
Qed.

Lemma choose_alias_in (tenv : Typing_env.t) (aliases : list variable)
    (c : variable) :
  Join_env.choose_alias tenv aliases = Some (Some c) -> c ∈ aliases.
Proof.
  unfold Join_env.choose_alias. destruct aliases as [ | var aliases];
    [ discriminate | ].
  destruct (Join_env.earliest_bound tenv var aliases) as [c' | ] eqn:Hc;
    [ | discriminate ].
  intros H. injection H as <-.
  destruct (earliest_bound_in tenv aliases var c' Hc) as [-> | Hin];
    [ left; reflexivity | right; exact Hin ].
Qed.

Lemma lookup_fold_cons {A : Type} (a : A) :
  ∀ (l : list variable) (m : finite_map A) (x : variable) (a' : A),
    lookup x (List.fold_left (fun m y => (y, a) :: m) l m) = Some a' ->
    (x ∈ l ∧ a' = a) ∨ lookup x m = Some a'.
Proof.
  induction l as [ | y l IH]; intros m x a' H; cbn [List.fold_left] in H.
  - right. exact H.
  - destruct (IH _ _ _ H) as [[Hin Heq] | Hlookup].
    + left. split; [ right; exact Hin | exact Heq ].
    + rewrite lookup_cons in Hlookup. destruct (Pos.eqb x y) eqn:Hxy.
      * apply Pos.eqb_eq in Hxy. subst y. injection Hlookup as <-.
        left. split; [ left; reflexivity | reflexivity ].
      * right. exact Hlookup.
Qed.

Lemma in_order_lookup {A : Type} (m : finite_map A) (x : variable) (a : A) :
  (x, a) ∈ rev (compact m) -> lookup x m = Some a.
Proof. intros H. apply in_rev in H. apply in_compact. exact H. Qed.

(* The meet environment after the fold: variables defined. *)
Lemma define_fold_env {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (s : Join_env.t) (tenv_of_aliases : Typing_env.t)
    (unique : variable -> bool) :
  ∀ (l : list (variable * flambda_kind)) (acc acc' : define_acc meet_env),
    List.fold_left (define_step ops s tenv_of_aliases unique) l (Some acc)
    = Some acc' ->
    ∃ kept : list (variable * flambda_kind),
      (∀ binding : variable * flambda_kind, binding ∈ kept -> binding ∈ l)
      ∧ fst (fst (fst (fst acc')))
        = variables_defined ops (fst (fst (fst (fst acc)))) kept.
Proof.
  induction l as [ | binding l IH]; intros acc acc' Hfold.
  - injection Hfold as <-. exists []. split; [ intros b [] | reflexivity ].
  - cbn [List.fold_left] in Hfold.
    destruct (define_step ops s tenv_of_aliases unique (Some acc) binding)
      as [acc1 | ] eqn:Hstep.
    2:{ rewrite (fold_left_none _ (fun b => eq_refl)) in Hfold.
        discriminate. }
    destruct (IH acc1 acc' Hfold) as [kept [Hkept Henv]].
    destruct acc as [[[[menv to_expand] equations] relations] definitions].
    cbn [define_step] in Hstep.
    destruct (Join_env.choose_alias _ _) as [[canon_var | ] | ];
      [ | | discriminate ].
    + destruct (Join_env.move_equation equations (fst binding) canon_var);
        [ | discriminate ].
      injection Hstep as <-. exists kept.
      split; [ intros b Hb; right; exact (Hkept b Hb) | exact Henv ].
    + destruct (unique (fst binding)).
      * injection Hstep as <-. exists kept.
        split; [ intros b Hb; right; exact (Hkept b Hb) | exact Henv ].
      * injection Hstep as <-. exists (binding :: kept). split.
        -- intros b [<- | Hb];
             [ left; reflexivity | right; exact (Hkept b Hb) ].
        -- rewrite Henv. destruct binding. reflexivity.
Qed.

(* ====================================================================== *)
(* What the fold keeps true at a run                                       *)
(* ====================================================================== *)

Section Define_run.

Context (source : Typing_env.t) (s : Join_env.t) (index : Index.t)
  (use : Typing_env.t) (statics : static_env) (valuation : env).

Hypothesis s_true : state_true source s.
Hypothesis s_kinded : state_kinded s.
Hypothesis envs_kinded : joined_envs_are typing_env_kinded s.
Hypothesis use_at : Join_env.joined_env s index = Some use.
Hypothesis use_satisfied : typing_env_satisfied_strictly statics valuation use.
Hypothesis use_tags : tags_at s use statics valuation.
Hypothesis aliases_valued : aliases_have_values s index use valuation.

Let tv : env := target_valuation s index use valuation.

Lemma alias_back (e x : variable) (v : value) :
  x
  ∈ Aliases_of_existentials.aliases_of_existential_var
      (Join_env.aliases_of_existentials s) e ->
  lookup x tv = Some v ->
  lookup e tv = Some v.
Proof.
  intros Hx Hv. destruct s_true as [_ [[_ [_ [Haliases _]]] _]].
  destruct (Haliases e x Hx) as [_ [_ [Hforward _]]].
  exact (Hforward index use statics valuation v use_at use_satisfied Hv).
Qed.

Lemma alias_value (e x : variable) (v : value) :
  x
  ∈ Aliases_of_existentials.aliases_of_existential_var
      (Join_env.aliases_of_existentials s) e ->
  lookup e tv = Some v ->
  lookup x tv = Some v.
Proof.
  intros Hx Hv. destruct (aliases_valued e x v Hx Hv) as [w Hw].
  pose proof (alias_back e x w Hx Hw) as Hw'.
  unfold tv in Hv, Hw'. rewrite Hv in Hw'. injection Hw' as <-. exact Hw.
Qed.

Lemma made_kind (e : variable) (kind : flambda_kind) (v : value) :
  (e, kind)
  ∈ Bindings_in_target_env.existential_variables_in_order (Join_env.bindings s)
  -> lookup e tv = Some v ->
  kind_of_value v = kind.
Proof.
  intros Hin Hv.
  exact (existential_value_kind s e kind index use statics valuation v
           (proj1 s_true) s_kinded envs_kinded (in_order_lookup _ _ _ Hin)
           use_at use_satisfied Hv).
Qed.

Definition true_at (x : variable) (ty : flambda_type) : Prop :=
  ∀ v : value, lookup x tv = Some v -> type_satisfied statics tv v ty.

Definition relation_true_at (r : variable) (pair : Relation.t * name) : Prop :=
  ∃ x : variable,
    snd pair = Name_var x
    ∧ ∀ v : value,
        lookup r tv = Some v ->
        ∃ (i : Z) (w : value),
          v = Value_const (Naked_immediate i)
          ∧ lookup x tv = Some w
          ∧ relation_holds statics (fst pair) w i.

Definition run_invariant {meet_env : Type} (acc : define_acc meet_env)
    : Prop :=
  let '(_, to_expand, equations, relations, definitions) := acc in
  (∀ (e : variable) (ty : flambda_type),
     lookup e to_expand = Some ty -> true_at e ty)
  ∧ (∀ (x : variable) (ty : flambda_type),
       lookup x equations = Some ty -> true_at x ty)
  ∧ (∀ (r : variable) (names : list (Relation.t * name)),
       lookup r relations = Some names ->
       names <> []
       ∧ ∀ pair : Relation.t * name, pair ∈ names -> relation_true_at r pair)
  ∧ definitions_hold definitions index valuation tv.

Lemma define_step_invariant {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool)
    (acc acc' : define_acc meet_env) (binding : variable * flambda_kind) :
  binding
  ∈ Bindings_in_target_env.existential_variables_in_order (Join_env.bindings s)
  -> run_invariant acc ->
  define_step ops s tenv_of_aliases unique (Some acc) binding = Some acc' ->
  run_invariant acc'.
Proof.
  intros Hin Hinv Hstep.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  destruct binding as [e kind]. cbn [define_step fst snd] in Hstep.
  destruct Hinv as [Hexpand [Hequations [Hrelations Hdefinitions]]].
  destruct (Join_env.choose_alias tenv_of_aliases
              (Aliases_of_existentials.aliases_of_existential_var
                 (Join_env.aliases_of_existentials s) e))
    as [[c | ] | ] eqn:Hchoose; [ | | discriminate ].
  - (* The variable has an alias in the target. *)
    pose proof (choose_alias_in _ _ _ Hchoose) as Hc.
    destruct (Join_env.move_equation equations e c) as [equations1 | ]
      eqn:Hmove; [ | discriminate ].
    injection Hstep as <-.
    assert (Halias : true_at e (alias_type_of kind (Var c))).
    { intros v Hv. apply alias_satisfied.
      split; [ exact (alias_value e c v Hc Hv) | ].
      exact (made_kind e kind v Hin Hv). }
    assert (Hequations1 :
              ∀ (x : variable) (ty : flambda_type),
                lookup x equations1 = Some ty -> true_at x ty).
    { unfold Join_env.move_equation in Hmove.
      destruct (is_bound c equations); [ discriminate | ].
      destruct (lookup e equations) as [ty_e | ] eqn:He.
      - injection Hmove as <-. intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x c) eqn:Hxc.
        + apply Pos.eqb_eq in Hxc. subst x. injection Hx as <-.
          intros v Hv. exact (Hequations e ty_e He v (alias_back e c v Hc Hv)).
        + rewrite lookup_unbind in Hx. destruct (Pos.eqb x e);
            [ discriminate | exact (Hequations x ty Hx) ].
      - injection Hmove as <-. exact Hequations. }
    split; [ | split; [ | split ] ].
    + intros x ty Hx. rewrite lookup_cons in Hx.
      destruct (Pos.eqb x e) eqn:Hxe; [ | exact (Hexpand x ty Hx) ].
      apply Pos.eqb_eq in Hxe. subst x. injection Hx as <-. exact Halias.
    + intros x ty Hx.
      destruct (lookup_fold_cons _ _ _ _ _ Hx) as [[Hdemoted ->] | Hold];
        [ | exact (Hequations1 x ty Hold) ].
      apply filter_In in Hdemoted. destruct Hdemoted as [Hxa _].
      intros v Hv. exact (Halias v (alias_back e x v Hxa Hv)).
    + intros r names Hr. unfold Join_env.move_inverse_relation in Hr.
      destruct (lookup e relations) as [names_e | ] eqn:He;
        [ | exact (Hrelations r names Hr) ].
      rewrite lookup_cons in Hr. destruct (Pos.eqb r c) eqn:Hrc.
      * apply Pos.eqb_eq in Hrc. subst r. injection Hr as <-.
        destruct (Hrelations e names_e He) as [Hnonempty Htrue]. split.
        -- intros Happ. apply app_eq_nil in Happ.
           exact (Hnonempty (proj1 Happ)).
        -- intros pair Hpair. apply in_app_or in Hpair.
           destruct Hpair as [Hpair | Hpair].
           ++ destruct (Htrue pair Hpair) as [x [Hx Hholds]]. exists x.
              split; [ exact Hx | ]. intros v Hv.
              exact (Hholds v (alias_back e c v Hc Hv)).
           ++ destruct (lookup c (unbind e relations)) as [existing | ]
                eqn:Hexisting; [ | destruct Hpair ].
              rewrite lookup_unbind in Hexisting.
              destruct (Pos.eqb c e); [ discriminate | ].
              exact (proj2 (Hrelations c existing Hexisting) pair Hpair).
      * rewrite lookup_unbind in Hr. destruct (Pos.eqb r e);
          [ discriminate | exact (Hrelations r names Hr) ].
    + intros x simples k simple v Hx Hfind Hvalue.
      unfold Join_env.move_definition in Hx.
      destruct (lookup e definitions) as [definition | ] eqn:He;
        [ | exact (Hdefinitions x simples k simple v Hx Hfind Hvalue) ].
      rewrite lookup_cons in Hx. destruct (Pos.eqb x c) eqn:Hxc.
      * apply Pos.eqb_eq in Hxc. subst x. injection Hx as ->.
        apply (alias_value e c v Hc).
        exact (Hdefinitions e simples k simple v He Hfind Hvalue).
      * rewrite lookup_unbind in Hx. destruct (Pos.eqb x e);
          [ discriminate
          | exact (Hdefinitions x simples k simple v Hx Hfind Hvalue) ].
  - (* The variable has no alias. *)
    destruct (unique e).
    + injection Hstep as <-. split; [ | split; [ | split ] ].
      * intros x ty Hx. rewrite lookup_cons in Hx.
        destruct (Pos.eqb x e) eqn:Hxe; [ | exact (Hexpand x ty Hx) ].
        apply Pos.eqb_eq in Hxe. subst x. injection Hx as <-.
        destruct (lookup e equations) as [ty_e | ] eqn:He;
          [ exact (Hequations e ty_e He) | ].
        intros v Hv. apply unknown_satisfied. exact (made_kind e kind v Hin Hv).
      * intros x ty Hx. rewrite lookup_unbind in Hx.
        destruct (Pos.eqb x e); [ discriminate | exact (Hequations x ty Hx) ].
      * exact Hrelations.
      * intros x simples k simple v Hx Hfind Hvalue.
        rewrite lookup_unbind in Hx. destruct (Pos.eqb x e);
          [ discriminate
          | exact (Hdefinitions x simples k simple v Hx Hfind Hvalue) ].
    + injection Hstep as <-.
      split; [ exact Hexpand | ]. split; [ exact Hequations | ].
      split; [ exact Hrelations | exact Hdefinitions ].
Qed.

Lemma define_fold_invariant {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (tenv_of_aliases : Typing_env.t) (unique : variable -> bool)
    (acc acc' : define_acc meet_env) :
  run_invariant acc ->
  List.fold_left (define_step ops s tenv_of_aliases unique)
    (Bindings_in_target_env.existential_variables_in_order
       (Join_env.bindings s))
    (Some acc)
  = Some acc' ->
  run_invariant acc'.
Proof.
  apply (fold_left_invariant (define_step ops s tenv_of_aliases unique)
           run_invariant (fun b => eq_refl)).
  intros a a' b Hb Ha Hstep.
  exact (define_step_invariant ops tenv_of_aliases unique a a' b Hb Ha Hstep).
Qed.

(* The invariant at the start. *)
Lemma define_start_invariant {meet_env : Type} (menv : meet_env)
    (relations : Join_env.inverse_relations) :
  relations_hold s relations ->
  relations_nonempty relations ->
  relation_scrutinees_have_values s relations index use valuation ->
  run_invariant
    (menv, [], Join_env.types_in_target_env s, relations,
     Join_env.definitions_of_existentials s).
Proof.
  intros Hrelations Hnonempty Hscrutinees.
  destruct s_true as [Hvalid [[Hequations _] _]].
  split; [ intros e ty He; discriminate | ]. split; [ | split ].
  - intros x ty Hx v Hv. destruct (Hequations x ty Hx) as [_ [_ Hholds]].
    exact (Hholds index use statics valuation v use_at use_satisfied use_tags
             Hv).
  - intros r names Hr. split; [ exact (Hnonempty r names Hr) | ].
    intros [relation n] Hpair.
    assert (Hrecorded : relation_recorded relations r relation n).
    { exists names. split; [ exact Hr | exact Hpair ]. }
    destruct (Hrelations r relation n Hrecorded) as [x [-> [_ [_ Hholds]]]].
    exists x. split; [ reflexivity | ]. intros v Hv.
    destruct (Hscrutinees r relation x v Hrecorded Hv) as [w Hw].
    destruct (Hholds index use statics valuation w use_at use_satisfied
                use_tags Hw) as [i [Hi Hrel]].
    exists i, w. unfold tv in Hv. rewrite Hv in Hi.
    injection Hi as ->. split; [ reflexivity | ].
    split; [ exact Hw | exact Hrel ].
  - destruct (target_valuation_extends_at s index use valuation Hvalid use_at)
      as [[use' [_ [_ Hdefinitions]]] _].
    exact Hdefinitions.
Qed.

(* What the equations made from the result of the fold say. *)
Lemma define_equations_true {meet_env : Type} (acc : define_acc meet_env)
    (fuel : nat) (equations' : list (name * flambda_type)) :
  run_invariant acc ->
  let '(_, to_expand, equations, _, _) := acc in
  traverse
    (fun binding =>
       let* ty :=
         project_variables_out (domain to_expand)
           (Join_env.expand_variable fuel to_expand) (snd binding)
       in
       Some (Name_var (fst binding), ty))
    (compact equations)
  = Some equations' ->
  equations_true_at statics tv equations'.
Proof.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  intros [Hexpand [Hequations _]] Htraverse n ty' v Hin Hv.
  destruct (traverse_in _ _ _ Htraverse _ Hin) as [[x ty] [Hx Hproject]].
  cbn [fst snd] in Hproject.
  destruct (project_variables_out (domain to_expand)
              (Join_env.expand_variable fuel to_expand) ty) as [ty0 | ]
    eqn:Hty; [ | discriminate ].
  injection Hproject as <- <-. apply in_compact in Hx.
  assert (Hexp :
            ∀ (e : variable) (t : flambda_type) (w : value),
              among e (domain to_expand) = true ->
              Join_env.expand_variable fuel to_expand e = Some t ->
              lookup e tv = Some w ->
              type_satisfied statics tv w t).
  { intros e t w _ He Hw.
    exact (expand_variable_true statics tv to_expand
             (fun e0 t0 w0 H0 Hw0 => Hexpand e0 t0 H0 w0 Hw0) fuel e t w He
             Hw). }
  destruct (project_all_sound statics tv (domain to_expand)
              (Join_env.expand_variable fuel to_expand) Hexp) as [Hsound _].
  exact (Hsound ty ty0 v Hty (Hequations x ty Hx v Hv)).
Qed.

Lemma define_relation_equations_true {meet_env : Type}
    (acc : define_acc meet_env) (fuel : nat)
    (equations' : list (name * flambda_type)) :
  run_invariant acc ->
  let '(_, to_expand, _, relations, _) := acc in
  traverse
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
    (compact relations)
  = Some equations' ->
  equations_true_at statics tv equations'.
Proof.
  destruct acc as [[[[menv to_expand] equations] relations] definitions].
  intros [Hexpand [_ [Hrelations _]]] Htraverse n ty' v Hin Hv.
  destruct (traverse_in _ _ _ Htraverse _ Hin) as [[r names] [Hr Hproject]].
  cbn [fst snd] in Hproject.
  match type of Hproject with
  | context [project_variables_out ?p ?e ?t] =>
    destruct (project_variables_out p e t) as [ty0 | ] eqn:Hty;
      [ | discriminate ]
  end.
  injection Hproject as <- <-. apply in_compact in Hr.
  assert (Hexp :
            ∀ (e : variable) (t : flambda_type) (w : value),
              among e (domain to_expand) = true ->
              Join_env.expand_variable fuel to_expand e = Some t ->
              lookup e tv = Some w ->
              type_satisfied statics tv w t).
  { intros e t w _ He Hw.
    exact (expand_variable_true statics tv to_expand
             (fun e0 t0 w0 H0 Hw0 => Hexpand e0 t0 H0 w0 Hw0) fuel e t w He
             Hw). }
  destruct (project_all_sound statics tv (domain to_expand)
              (Join_env.expand_variable fuel to_expand) Hexp) as [Hsound _].
  apply (Hsound _ ty0 v Hty). rewrite type_satisfied_naked_immediate.
  cbn [descr_satisfied].
  destruct (Hrelations r names Hr) as [Hnonempty Htrue].
  destruct names as [ | pair0 names]; [ exfalso; exact (Hnonempty eq_refl) | ].
  destruct (Htrue pair0 (or_introl eq_refl)) as [x0 [_ Hholds0]].
  cbn in Hv.
  destruct (Hholds0 v Hv) as [i [w0 [Hi _]]].
  exists i. split; [ exact Hi | ]. split; [ exact I | ].
  intros relation n Hpair. cbn [inverse_relations] in Hpair.
  destruct (Htrue (relation, n) Hpair) as [x [Hx Hholds]].
  cbn [fst snd] in Hx, Hholds. subst n.
  destruct (Hholds v Hv) as [i' [w [Hi' [Hw Hrel]]]].
  rewrite Hi in Hi'. injection Hi' as <-.
  exists w. split; [ exact Hw | exact Hrel ].
Qed.

End Define_run.

(* ---------------------------------------------------------------------- *)
(* The aliases have values                                                 *)
(* ---------------------------------------------------------------------- *)

Theorem aliases_have_values_of_values
  : aliases_have_values_of_values_statement.
Proof.
  intros source s index use valuation [Hvalid [Hholds _]] Huse Himported
    Hjoined e x v Hx Hv.
  destruct Hholds as [_ [_ [Haliases _]]].
  destruct (Haliases e x Hx)
    as [Hexists [Hexistential [_ [Hlocated Hlevels]]]].
  pose proof Hvalid as [_ [Hnew [[_ [_ Hdefined]] _]]].
  pose proof (Hdefined e Hexistential) as Hbound.
  apply is_bound_lookup in Hbound.
  destruct Hbound as [[simples kind] Hdefinition].
  destruct (existential_value s e simples kind index use valuation v Hvalid
              Hdefinition Huse Hv) as [simple [Hfind _]].
  destruct (Hlocated simples kind index simple Hdefinition Hfind)
    as [use' [Huse' Hknows]].
  rewrite Huse in Huse'. injection Huse' as <-.
  assert (Hknown : known_to s use x = true).
  { unfold known_to. unfold knows in Hknows. rewrite Hknows. reflexivity. }
  rewrite (target_valuation_known s index use valuation x Hvalid Huse Hknown).
  destruct (Typing_env.mem None (Join_env.source_env s) (Name_var x))
    eqn:Hsource.
  - exact (Hjoined x Hsource (Hlevels Hsource)).
  - apply (Himported x); [ | exact Hknows ].
    unfold exists_in_target_env, kind_in_target_env in Hexists.
    unfold Join_env.source_env, Typing_env.mem in Hsource.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types
                   (Bindings_in_target_env.source_env (Join_env.bindings s))))
      as [[ty btm] | ]; [ discriminate | ].
    destruct (lookup x
                (Bindings_in_target_env.existential_variables
                   (Join_env.bindings s))) as [k | ] eqn:Hexistential_x.
    + exfalso.
      assert (Hx_made : is_existential s x).
      { unfold is_existential. apply is_bound_lookup. exists k.
        exact Hexistential_x. }
      exact (proj2 (Hnew x Hx_made) index use Huse Hknows).
    + unfold is_imported. apply is_bound_lookup.
      destruct (lookup x
                  (Bindings_in_target_env.imported_variables
                     (Join_env.bindings s))) as [k | ]; [ | discriminate ].
      exists k. reflexivity.
Qed.

(* ---------------------------------------------------------------------- *)
(* The relations kept by the rounds                                        *)
(* ---------------------------------------------------------------------- *)

Lemma add_relation_nonempty (relations : Join_env.inverse_relations)
    (var : variable) (relation : Relation.t) (scrutinee : name) :
  relations_nonempty relations ->
  relations_nonempty
    (Join_env.add_to_inverse_relations relations var relation scrutinee).
Proof.
  intros H r names Hr. unfold Join_env.add_to_inverse_relations in Hr.
  rewrite lookup_cons in Hr. destruct (Pos.eqb r var).
  - injection Hr as <-. discriminate.
  - exact (H r names Hr).
Qed.

Lemma recover_nonempty (relations relations' : Join_env.inverse_relations)
    (n : name) (ty ty' : flambda_type) :
  Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
  ->
  relations_nonempty relations ->
  relations_nonempty relations'.
Proof.
  intros Hrecover H.
  destruct (recover_relations_cases _ _ _ _ _ Hrecover)
    as [-> | [Hcase | [Hcase | Hcase]]]; [ exact H | | | ].
  - destruct Hcase as [is_int [imm_ty [g [blocks [ext [uniq [_ [_ ->]]]]]]]].
    exact (add_relation_nonempty _ _ _ _ H).
  - destruct Hcase as [r [imm [get_tag [blocks [ext [uniq [_ ->]]]]]]].
    exact (add_relation_nonempty _ _ _ _ H).
  - destruct Hcase as [non_null [r [_ ->]]].
    exact (add_relation_nonempty _ _ _ _ H).
Qed.

Lemma round_nonempty (choices : Join_env.choices)
    (jt : Join_env.n_way_join_type) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  relations_nonempty relations ->
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  relations_nonempty relations'.
Proof.
  intros Hnonempty Hround. unfold Join_env.n_way_join_round in Hround.
  refine (fold_left_invariant _
            (fun a : Join_env.inverse_relations * Join_env.t =>
               relations_nonempty (fst a))
            (fun b => eq_refl) _ _ (relations, s) (relations', s') Hnonempty
            Hround).
  intros [relations0 s0] [relations1 s1] [var types] _ H0 Hstep.
  cbn [fst snd] in Hstep, H0 |- *.
  destruct (jt s0 (Index.bindings types)) as [[joined s2] | ];
    [ | discriminate ].
  destruct joined as [ty | ].
  - destruct (Join_env.recover_relations_on choices s2 var types).
    + destruct (Join_env.recover_inverse_relations relations0 (Name_var var)
                  ty) as [[ty' relations2] | ] eqn:Hrecover;
        [ | discriminate ].
      destruct (Join_env.add_equation_in_target_env s2 var ty');
        [ | discriminate ].
      injection Hstep as <- _. exact (recover_nonempty _ _ _ _ _ Hrecover H0).
    + destruct (Join_env.add_equation_in_target_env s2 var ty);
        [ | discriminate ].
      injection Hstep as <- _. exact H0.
  - injection Hstep as <- _. exact H0.
Qed.

Theorem delayed_equations_relations_nonempty
    : delayed_equations_relations_nonempty_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH];
    intros choices jt s s' relations relations' equations Hnonempty Hloop;
    [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as <- _. exact Hnonempty.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    exact (IH choices jt s1 s' relations1 relations' _
             (round_nonempty _ _ _ _ _ _ _ Hnonempty Hround) Hloop).
Qed.

Theorem loop_relations_nonempty : loop_relations_nonempty_statement.
Proof.
  intros depth. induction depth as [ | depth IH];
    intros p fuel choices jt s s' relations relations' existentials Hnonempty
      Hloop; cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as <- _. exact Hnonempty.
  - destruct (Keyed.is_empty existentials).
    + injection Hloop as <- _. exact Hnonempty.
    + destruct (Join_env.equations_of_existentials p s (compact existentials))
        as [equations | ]; [ | discriminate ].
      destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                  relations equations) as [[relations1 s1] | ] eqn:Hdelayed;
        [ | discriminate ].
      exact (IH p fuel choices jt s1 s' relations1 relations' _
               (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                  Hnonempty Hdelayed)
               Hloop).
Qed.

Theorem expand_variable_sound : expand_variable_sound_statement.
Proof.
  intros statics vars to_expand fuel e ty w Hbase.
  exact (expand_variable_true statics vars to_expand Hbase fuel e ty w).
Qed.

Theorem equations_true_with_values : equations_true_with_values_statement.
Proof.
  intros statics vars equations Htrue Hvalues.
  apply env_extension_lookup_iff. intros n ty Hlookup.
  pose proof (keyed_lookup_in name_eqb name_eqb_eq n ty equations Hlookup)
    as Hin.
  destruct (Hvalues n ty Hin) as [v Hv]. exists v.
  split; [ exact Hv | exact (Htrue n ty v Hin Hv) ].
Qed.

Theorem define_variables_true : define_variables_true_statement.
Proof.
  intros meet_env meet_expanded_head ops fuel mh source s menv target
    relations definitions Htrue Hkinded Henvs Hrelations Hnonempty Hdefine.
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
  destruct (define_fold_env ops s _ unique _ _ _ Hfold)
    as [kept [Hkept Hmenv1]].
  unfold start in Hmenv1. cbn [fst] in Hmenv1.
  exists kept, equations', relation_equations, menv2.
  split; [ exact Hkept | ]. split.
  { unfold variables_defined. rewrite fold_left_app.
    unfold variables_defined in Hmenv1. rewrite <- Hmenv1. exact Hadd1. }
  split; [ exact Hadd2 | ].
  intros index use statics valuation Huse Hsatisfied Htags Haliases
    Hscrutinees.
  pose proof (define_start_invariant source s index use statics valuation
                Htrue Huse Hsatisfied Htags
                (List.fold_left
                   (fun target_env binding =>
                      Join_env.add_variable_definition ops target_env
                        (fst binding) (snd binding) Name_mode.in_types)
                   (Bindings_in_target_env.imported_variables_in_order
                      (Join_env.bindings s))
                   menv)
                relations Hrelations Hnonempty Hscrutinees) as Hstart.
  pose proof (define_fold_invariant source s index use statics valuation Htrue
                Hkinded Henvs Huse Hsatisfied Haliases ops _ unique _ _ Hstart
                Hfold) as Hinvariant.
  split; [ | split ].
  - exact (define_equations_true s index use statics valuation _ fuel
             equations' Hinvariant Hequations).
  - exact (define_relation_equations_true s index use statics valuation _
             fuel relation_equations Hinvariant Hrelation_equations).
  - exact (proj2 (proj2 (proj2 Hinvariant))).
Qed.

Abbreviation envs_invariant := (joined_envs_are typing_env_invariant).
Abbreviation envs_kinded := (joined_envs_are typing_env_kinded).

(* ---------------------------------------------------------------------- *)
(* The first phases keep the state kinded                                  *)
(* ---------------------------------------------------------------------- *)

Lemma import_or_delay_kinded (p : Join_env.parameters) (fuel : nat)
    (s s' : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type) :
  fresh_avoids p ->
  state_valid s ->
  state_kinded s ->
  envs_invariant s ->
  envs_kinded s ->
  types_hold s var types ->
  (∀ (index index' : Index.t) (t t' : flambda_type),
     Keyed.lookup Index.equal index types = Some t ->
     Keyed.lookup Index.equal index' types = Some t' ->
     FlambdaTypeGrammar.kind t = FlambdaTypeGrammar.kind t') ->
  Join_env.import_or_delay_n_way_join_type p fuel s var types = Some s' ->
  state_kinded s'.
Proof.
  intros Hfresh Hvalid Hstate Hinvariant Hkinded Htypes Hkinds Himport.
  unfold Join_env.import_or_delay_n_way_join_type,
    Join_env.import_or_delay_n_way_join_type_with in Himport.
  destruct (Index.choose types) as [[index0 any_ty] | ] eqn:Hchoose;
    [ | discriminate ].
  destruct
    (Joined_envs.get_alias_then_canonical_simple_ignoring_name_mode_exn
       (Join_env.joined_envs s) types)
    as [[canonicals | ] | ] eqn:Hcanonicals; [ | | discriminate ].
  - apply (add_equals_keeps_kinded p _ s s' (FlambdaTypeGrammar.kind any_ty)
             var canonicals Hfresh (import_var_keeps p fuel Hfresh)
             (import_var_keeps_kinded p fuel Hfresh) Hvalid Hstate Hinvariant
             Hkinded); [ | exact Himport ].
    intros index c use Hfind Huse.
    apply (keyed_lookup_in Index.equal index_equal_decides) in Hfind.
    destruct (canonicals_of_alias_types_spec _ _ _ Hcanonicals _ _ Hfind)
      as [ty [env [alias [Hin [Hnth [Halias ->]]]]]].
    change (Join_env.joined_env s index = Some env) in Hnth.
    rewrite Huse in Hnth. injection Hnth as <-.
    apply (keyed_in_compact Index.equal index_equal_decides) in Hin.
    destruct (proj1 (proj2 Htypes) index ty Hin) as [use' [Huse' [_ Hty]]].
    rewrite Huse in Huse'. injection Huse' as <-.
    apply (canonical_kinded use alias _ (Hinvariant _ _ Huse)
             (Hkinded _ _ Huse)).
    rewrite <- (Hkinds index index0 ty any_ty Hin
                  (choose_lookup _ _ _ Hchoose)).
    exact (Hty alias _ (get_alias_in_aliases _ _ Halias)).
  - exact (concrete_keeps_kinded p _ s s' var types
             (import_var_keeps p fuel Hfresh)
             (import_var_keeps_kinded p fuel Hfresh) Hvalid Hstate Hinvariant
             Hkinded Himport).
Qed.

Lemma join_aliases_fold_kinded (p : Join_env.parameters) (fuel : nat)
    (s0 : Join_env.t) (envs : Joined_envs.t) (exists0 : variable -> bool) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  envs_invariant s0 ->
  envs_kinded s0 ->
  ∀ (names : list name) (s s' : Join_env.t),
    (∀ x : variable,
       Name_var x ∈ names ->
       exists0 x = true ->
       types_hold s0 x (Join_env.types_of_name_in_levels envs (Name_var x))
       ∧ (∀ (index index' : Index.t) (t t' : flambda_type),
            Keyed.lookup Index.equal index
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t ->
            Keyed.lookup Index.equal index'
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t' ->
            FlambdaTypeGrammar.kind t = FlambdaTypeGrammar.kind t')
       ∧ (∀ (index : Index.t) (t : flambda_type) (use : Typing_env.t),
            Keyed.lookup Index.equal index
              (Join_env.types_of_name_in_levels envs (Name_var x))
            = Some t ->
            Join_env.joined_env s0 index = Some use ->
            knows use x)
       ∧ (knows (Join_env.source_env s0) x -> joined_everywhere s0 x)) ->
    later s0 s ->
    same_envs s0 s ->
    state_holds s ->
    state_kinded s ->
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
    state_kinded s'.
Proof.
  intros Hfresh Heqb Hinvariant0 Hkinded0.
  induction names as [ | n names IH];
    intros s s' Hnames Hlater Hsame Hholds Hstate Hfold.
  - injection Hfold as <-. exact Hstate.
  - cbn [List.fold_left] in Hfold.
    pose proof (fun x Hx => Hnames x (or_intror Hx)) as Hrest.
    destruct n as [x | sym];
      [ | exact (IH s s' Hrest Hlater Hsame Hholds Hstate Hfold) ].
    destruct (exists0 x) eqn:Hexists;
      [ | exact (IH s s' Hrest Hlater Hsame Hholds Hstate Hfold) ].
    destruct (Join_env.import_or_delay_n_way_join_type p fuel s x
                (Join_env.types_of_name_in_levels envs (Name_var x)))
      as [s1 | ] eqn:Hs1.
    + destruct (Hnames x (or_introl eq_refl) Hexists)
        as [Htypes [Hkinds [Hknown Hlevels]]].
      pose proof (later_envs_invariant _ _ Hlater Hinvariant0) as Hinvariant.
      pose proof (later_envs_kinded _ _ Hlater Hkinded0) as Hkinded.
      pose proof (later_types_hold _ _ _ _ Hlater Htypes) as Htypes_s.
      assert (Hknown_s :
                ∀ (index : Index.t) (t : flambda_type) (use : Typing_env.t),
                  Keyed.lookup Index.equal index
                    (Join_env.types_of_name_in_levels envs (Name_var x))
                  = Some t ->
                  Join_env.joined_env s index = Some use ->
                  knows use x).
      { intros index t use Ht Huse. rewrite (later_env _ _ _ Hlater) in Huse.
        exact (Hknown index t use Ht Huse). }
      assert (Hlevels_s :
                knows (Join_env.source_env s) x -> joined_everywhere s x).
      { intros Hk index use equations Hlookup.
        rewrite (proj2 (proj2 (proj2 Hlater))) in Hk.
        rewrite Hsame in Hlookup.
        exact (Hlevels Hk index use equations Hlookup). }
      destruct (import_or_delay_holds p fuel s s1 x _ Hfresh Heqb
                  (proj1 (proj2 Hlater)) Hinvariant Hkinded Hholds Htypes_s
                  Hkinds Hknown_s Hlevels_s Hs1) as [Hlater1 Hholds1].
      exact (IH s1 s' Hrest (later_trans _ _ _ Hlater Hlater1)
               (same_envs_trans _ _ _ Hsame
                  (import_or_delay_same_envs _ _ _ _ _ _ Hs1))
               Hholds1
               (import_or_delay_kinded p fuel s s1 x _ Hfresh
                  (proj1 (proj2 Hlater)) Hstate Hinvariant Hkinded Htypes_s
                  Hkinds Hs1)
               Hfold).
    + exfalso. clear -Hfold.
      induction names as [ | n' names IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Lemma join_aliases_kinded (p : Join_env.parameters) (fuel : nat)
    (s0 s' : Join_env.t) :
  fresh_avoids p ->
  parameters_type_eqb_sound p ->
  state_valid s0 ->
  state_kinded s0 ->
  envs_invariant s0 ->
  envs_kinded s0 ->
  state_holds s0 ->
  levels_found (Join_env.joined_envs s0) ->
  (∀ x : variable,
     exists_in_target_env kind_in_target_env s0 x = true ->
     knows (Join_env.source_env s0) x) ->
  Join_env.join_aliases_into_bindings p fuel s0 (Join_env.joined_envs s0)
  = Some s' ->
  state_kinded s'.
Proof.
  intros Hfresh Heqb Hvalid0 Hstate0 Hinvariant0 Hkinded0 Hholds0 Hlevels
    Hsource Hjoin.
  apply (join_aliases_fold_kinded p fuel s0 (Join_env.joined_envs s0)
           (Bindings_in_target_env.exists_in_target_env (Join_env.bindings s0))
           Hfresh Heqb Hinvariant0 Hkinded0
           (Join_env.names_with_equations_everywhere (Join_env.joined_envs s0))
           s0 s');
    [ | exact (later_refl s0 Hvalid0) | reflexivity | exact Hholds0
      | exact Hstate0 | exact Hjoin ].
  intros x Hx Hexists. rewrite <- kind_in_target_env_exists in Hexists.
  pose proof (Hsource x Hexists) as Hknown.
  split; [ | split; [ | split ] ].
  - exact (level_types_hold s0 x Hvalid0 Hinvariant0 Hkinded0 Hlevels Hknown
             Hx).
  - exact (level_types_one_kind s0 x Hstate0 Hlevels Hknown).
  - intros index t use Ht Huse. rewrite types_of_name_lookup in Ht.
    destruct (Keyed.lookup Index.equal index (Join_env.joined_envs s0))
      as [[use0 equations] | ] eqn:Hindex; [ | discriminate ].
    destruct (Hlevels _ _ _ _ _ Hindex Ht) as [btm Hfound].
    assert (Huse0 : Join_env.joined_env s0 index = Some use0).
    { apply joined_env_lookup. exists equations. exact Hindex. }
    rewrite Huse in Huse0. injection Huse0 as <-.
    apply bound_knows. unfold Keyed.is_bound. rewrite Hfound. reflexivity.
  - intros _ index use equations Hindex.
    exact (names_everywhere_bound _ _ Hx _ _ _ Hindex).
Qed.

Section Rounds_kinded.

Context (source : Typing_env.t) (jt : Join_env.n_way_join_type)
  (choices : Join_env.choices).

Hypothesis jt_sound :
  FlambdaJoinSpec.n_way_join_type_sound typing_env_invariant
    Join_env.joined_env Join_env.definitions_of_existentials
    kind_in_target_env (state_true source) state_kinded jt.

Lemma join_call_kinded (s s0 : Join_env.t) (var : variable)
    (types : Keyed.finite_map Index.t flambda_type)
    (joined : Or_unknown.t flambda_type) :
  state_true source s ->
  envs_invariant s ->
  envs_kinded s ->
  types_hold s var types ->
  state_kinded s ->
  jt s (Index.bindings types) = Some (joined, s0) ->
  state_kinded s0.
Proof.
  intros Htrue Hinvariant Hkinded [Hvar [Hfound _]] Hstate Hjoin.
  assert (Hscope :
            join_args_in_scope Join_env.joined_env type_in_scope s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [Hscope _]]].
    exists use. split; [ exact Huse | exact Hscope ]. }
  assert (Hkinds :
            join_args_in_scope Join_env.joined_env type_kinded s
              (Index.bindings types)).
  { intros index t Hin. apply bindings_lookup in Hin.
    destruct (Hfound _ _ Hin) as [use [Huse [_ Hkind]]].
    exists use. split; [ exact Huse | exact Hkind ]. }
  destruct (jt_sound s s0 (Index.bindings types) joined Htrue Hinvariant
              Hkinded (bindings_distinct types) Hscope Hkinds Hjoin)
    as [_ [_ [_ Hkinded0]]].
  exact (proj1 (Hkinded0 Hstate)).
Qed.

Lemma round_fold_kinded :
  ∀ (bindings : list (variable * Keyed.finite_map Index.t flambda_type))
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       In (var, types) bindings -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    List.fold_left (round_step jt choices) bindings (Some (relations, s))
    = Some (relations', s') ->
    state_kinded s'.
Proof.
  induction bindings as [ | [var types] bindings IH];
    intros s s' relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hstate Hfold.
  - cbn in Hfold. injection Hfold as _ <-. exact Hstate.
  - cbn [List.fold_left] in Hfold.
    destruct (round_step jt choices (Some (relations, s)) (var, types))
      as [[relations1 s1] | ] eqn:Hstep;
      [ | rewrite round_step_none in Hfold; discriminate ].
    destruct (round_fold_holds source jt choices jt_sound [(var, types)] s s1
                relations relations1
                (fun v t Hin =>
                   Htypes v t
                     (or_introl
                        (match Hin with
                         | or_introl H => H
                         | or_intror H => match H with end
                         end)))
                Htrue Hinvariant Hkinded Hrelations Hstep)
      as [Hlater1 [Hholds1 Hrelations1]].
    assert (Hstate1 : state_kinded s1).
    { unfold round_step in Hstep. cbn [fst snd] in Hstep.
      destruct (jt s (Index.bindings types)) as [[joined s0] | ] eqn:Hjoin;
        [ | discriminate ].
      pose proof (join_call_kinded s s0 var types joined Htrue Hinvariant
                    Hkinded (Htypes var types (or_introl eq_refl)) Hstate
                    Hjoin) as Hstate0.
      destruct joined as [ty | ].
      - destruct (Join_env.recover_relations_on choices s0 var types).
        + destruct (Join_env.recover_inverse_relations relations
                      (Name_var var) ty)
            as [[ty' relations0] | ]; [ | discriminate ].
          destruct (Join_env.add_equation_in_target_env s0 var ty')
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as _ <-.
          exact (add_equation_keeps_kinded _ _ _ _ Hstate0 Hadd).
        + destruct (Join_env.add_equation_in_target_env s0 var ty)
            as [s2 | ] eqn:Hadd; [ | discriminate ].
          injection Hstep as _ <-.
          exact (add_equation_keeps_kinded _ _ _ _ Hstate0 Hadd).
      - injection Hstep as _ <-. exact Hstate0. }
    exact (IH s1 s' relations1 relations'
             (fun v t Hin =>
                later_types_hold _ _ _ _ Hlater1 (Htypes v t (or_intror Hin)))
             (later_true source s s1 Hlater1 Hholds1 (proj2 (proj2 Htrue)))
             (later_envs_invariant _ _ Hlater1 Hinvariant)
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hstate1
             Hfold).
Qed.

Lemma delayed_equations_kinded :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types -> types_hold s var types) ->
    state_true source s ->
    envs_invariant s ->
    envs_kinded s ->
    relations_hold s relations ->
    state_kinded s ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    state_kinded s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' equations relations relations' Htypes Htrue Hinvariant Hkinded
      Hrelations Hstate Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. exact Hstate.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    destruct (round_holds source jt choices jt_sound s s1 equations relations
                relations1 Htypes Htrue Hinvariant Hkinded Hrelations Hround)
      as [Hlater1 [Hholds1 Hrelations1]].
    assert (Hstate1 : state_kinded s1).
    { apply (round_fold_kinded (compact equations) s s1 relations relations1);
        [ | exact Htrue | exact Hinvariant | exact Hkinded | exact Hrelations
          | exact Hstate | exact Hround ].
      intros var types Hin. apply Htypes. apply in_compact. exact Hin. }
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
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hstate1
             Hloop).
Qed.

Lemma loop_kinded (p : Join_env.parameters) :
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
    state_kinded s ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    state_kinded s'.
Proof.
  intros Hconst.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hdefinitions Htrue
      Hinvariant Hkinded Hrelations Hstate Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. exact Hstate.
  - destruct (Keyed.is_empty existentials).
    { injection Hloop as _ <-. exact Hstate. }
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
                Hrelations Hdelayed) as [Hlater1 [Hholds1 Hrelations1]].
    pose proof (delayed_equations_kinded fuel s s1 equations relations
                  relations1 Htypes Htrue Hinvariant Hkinded Hrelations Hstate
                  Hdelayed) as Hstate1.
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
             (later_envs_kinded _ _ Hlater1 Hkinded) Hrelations1 Hstate1
             Hloop).
Qed.

End Rounds_kinded.

Theorem first_phases_kinded : first_phases_kinded_statement.
Proof.
  intros source jt choices p fuel avoid envs s1 s2 s3 relations2 relations3
    Hjt Hfresh Heqb Hconst Hsource Huses Hinvariant Hkinded Hlevels Hkinds
    Haliases Hdelayed Hloop.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source)) in *.
  pose proof (created_state_true avoid source envs Hsource Huses) as Htrue0.
  fold s0 in Htrue0.
  pose proof (created_state_kinded avoid source envs Hkinds) as Hstate0.
  fold s0 in Hstate0.
  assert (Hknown0 :
            ∀ x : variable,
              exists_in_target_env kind_in_target_env s0 x = true ->
              knows source x).
  { intros x Hx. rewrite kind_in_target_env_exists in Hx.
    unfold s0, Join_env.create, Bindings_in_target_env.exists_in_target_env,
      Bindings_in_target_env.from_source_env in Hx.
    cbn in Hx. rewrite !orb_false_r in Hx. exact Hx. }
  assert (Hinvariant0 : envs_invariant s0) by exact Hinvariant.
  assert (Hkinded0 : envs_kinded s0) by exact Hkinded.
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
  pose proof (later_envs_invariant _ _ Hlater1 Hinvariant0) as Hinvariant1.
  pose proof (later_envs_kinded _ _ Hlater1 Hkinded0) as Hkinded1.
  destruct (delayed_equations_hold source jt choices Hjt fuel s1 s2 _ []
              relations2 Htypes1 Htrue1 Hinvariant1 Hkinded1 Hrelations1
              Hdelayed) as [Hlater2 [Hholds2 Hrelations2]].
  pose proof (delayed_equations_kinded source jt choices Hjt fuel s1 s2 _ []
                relations2 Htypes1 Htrue1 Hinvariant1 Hkinded1 Hrelations1
                Hstate1 Hdelayed) as Hstate2.
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
  exact (loop_kinded source jt choices Hjt p Hconst (Join_env.join_depth p)
           fuel s2 s3 relations2 relations3 _ Hdefinitions2
           (later_true source s1 s2 Hlater2 Hholds2 (proj2 (proj2 Htrue1)))
           (later_envs_invariant _ _ Hlater02 Hinvariant0)
           (later_envs_kinded _ _ Hlater02 Hkinded0) Hrelations2 Hstate2
           Hloop).
Qed.

(* ---------------------------------------------------------------------- *)
(* The join, up to what it adds to the target                              *)
(* ---------------------------------------------------------------------- *)

Theorem cut_and_n_way_join0_true : cut_and_n_way_join0_true_statement.
Proof.
  intros meet_env meet_expanded_head ops p fuel choices jt mh avoid menv
    target source envs definitions names Hjt Hfresh Heqb Hconst Hsource Huses
    Hinvariant Hkinded Hlevels Hkinds Hempty Hjoin.
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
  pose proof (first_phases_kinded source jt choices p fuel avoid envs s1 s2 s3
                relations2 relations3 Hjt Hfresh Heqb Hconst Hsource Huses
                Hinvariant Hkinded Hlevels Hkinds Haliases Hdelayed Hloop)
    as Hstate3.
  assert (Hnonempty1 : relations_nonempty []).
  { intros r names' Hr. discriminate Hr. }
  pose proof (loop_relations_nonempty _ _ _ _ _ _ _ _ _ _
                (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                   Hnonempty1 Hdelayed)
                Hloop) as Hnonempty3.
  assert (Hkinded3 : joined_envs_are typing_env_kinded s3).
  { intros index use Huse. rewrite (proj1 Hgrows3) in Huse.
    exact (Hkinded index use Huse). }
  destruct (define_variables_true meet_env meet_expanded_head ops fuel mh
              source s3 _ target' relations3 definitions' Htrue3 Hstate3
              Hkinded3 Hrelations3 Hnonempty3 Hdefine)
    as [kept [equations [relation_equations [menv' [Hkept [Hadd1 [Hadd2
          Hruns]]]]]]].
  exists s3, relations3, kept, equations, relation_equations, menv'.
  split; [ exact Htrue3 | ]. split; [ exact Hstate3 | ].
  split; [ exact Hrelations3 | ]. split; [ exact Hgrows3 | ].
  split; [ reflexivity | ]. split; [ exact Hkept | ].
  split; [ exact Hadd1 | ]. split; [ exact Hadd2 | exact Hruns ].
Qed.

(* ---------------------------------------------------------------------- *)
(* The joined environments of the inputs                                   *)
(* ---------------------------------------------------------------------- *)

Lemma combine_seq_in {A : Type} :
  ∀ (xs : list A) (start i : nat) (x : A),
    (i, x) ∈ combine (seq start (length xs)) xs ->
    ∃ k : nat, i = start + k ∧ nth_error xs k = Some x.
Proof.
  induction xs as [ | x0 xs IH]; intros start i x Hin; cbn in Hin;
    [ destruct Hin | ].
  destruct Hin as [Heq | Hin].
  - injection Heq as <- <-. exists 0. split; [ lia | reflexivity ].
  - destruct (IH (S start) i x Hin) as [k [-> Hk]]. exists (S k).
    split; [ lia | exact Hk ].
Qed.

Lemma in_combine_seq {A : Type} :
  ∀ (xs : list A) (start k : nat) (x : A),
    nth_error xs k = Some x ->
    (start + k, x) ∈ combine (seq start (length xs)) xs.
Proof.
  induction xs as [ | x0 xs IH]; intros start k x Hk;
    [ destruct k; discriminate | ].
  cbn. destruct k as [ | k].
  - injection Hk as <-. left. rewrite Nat.add_0_r. reflexivity.
  - right. replace (start + S k) with (S start + k) by lia.
    exact (IH (S start) k x Hk).
Qed.

Lemma combine_seq_distinct {A : Type} :
  ∀ (xs : list A) (start : nat),
    NoDup (map fst (combine (seq start (length xs)) xs)).
Proof.
  induction xs as [ | x0 xs IH]; intros start; cbn; [ constructor | ].
  constructor; [ | exact (IH (S start)) ].
  intros Hin. apply in_map_iff in Hin. destruct Hin as [[i x] [Hi Hin]].
  cbn in Hi. subst i. destruct (combine_seq_in xs (S start) start x Hin)
    as [k [Hk _]]. lia.
Qed.

Lemma number_in {A : Type} (xs : list A) (index : Index.t) (x : A) :
  (index, x) ∈ Index.number xs <-> nth_error xs index = Some x.
Proof.
  unfold Index.number. split.
  - intros Hin. destruct (combine_seq_in xs 0 index x Hin) as [k [-> Hk]].
    exact Hk.
  - intros Hk. exact (in_combine_seq xs 0 index x Hk).
Qed.

(* The uses, numbered, without their identifiers. *)
Definition numbered_uses {A : Type} (inputs : join_inputs A)
    : list (Index.t * Typing_env.t) :=
  map (fun binding => (fst binding, snd (snd binding)))
    (Index.number (uses inputs)).

Lemma numbered_uses_in {A : Type} (inputs : join_inputs A) (index : Index.t)
    (use : Typing_env.t) :
  (index, use) ∈ numbered_uses inputs <-> use_at inputs index use.
Proof.
  unfold numbered_uses, use_at. rewrite in_map_iff. split.
  - intros [[i [id u]] [Heq Hin]]. cbn in Heq. injection Heq as -> ->.
    exists id. apply number_in. exact Hin.
  - intros [id Hnth]. exists (index, (id, use)).
    split; [ reflexivity | apply number_in; exact Hnth ].
Qed.

Lemma numbered_uses_distinct {A : Type} (inputs : join_inputs A) :
  NoDup (map fst (numbered_uses inputs)).
Proof.
  unfold numbered_uses. rewrite map_map. cbn [fst].
  exact (combine_seq_distinct (uses inputs) 0).
Qed.

Lemma cut_for_joins_found (cut_after : Scope.t) :
  ∀ (envs : list (Index.t * Typing_env.t)) (joined : Joined_envs.t),
    Join_env.cut_for_joins cut_after envs = Some joined ->
    ∀ (index : Index.t) (use : Typing_env.t)
      (equations : Keyed.finite_map name flambda_type),
      Keyed.lookup Index.equal index joined = Some (use, equations) ->
      (index, use) ∈ envs
      ∧ Typing_env.is_bottom use = false
      ∧ ∃ level : Typing_env_level.t,
          Typing_env.cut use cut_after = Some level
          ∧ equations = Typing_env_level.equations level.
Proof.
  induction envs as [ | [i tenv] envs IH];
    intros joined Hcut index use equations Hlookup;
    cbn [Join_env.cut_for_joins] in Hcut.
  - injection Hcut as <-. discriminate.
  - destruct (Join_env.cut_for_joins cut_after envs) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    destruct (Typing_env.is_bottom tenv) eqn:Hbottom.
    + injection Hcut as <-.
      destruct (IH rest eq_refl index use equations Hlookup) as [Hin H].
      split; [ right; exact Hin | exact H ].
    + destruct (Typing_env.cut tenv cut_after) as [level | ] eqn:Hlevel;
        [ | discriminate ].
      injection Hcut as <-. rewrite keyed_lookup_cons in Hlookup.
      destruct (Index.equal index i) eqn:Heq.
      * injection Hlookup as <- <-. apply index_equal_decides in Heq.
        subst i. split; [ left; reflexivity | ].
        split; [ exact Hbottom | ].
        exists level. split; [ exact Hlevel | reflexivity ].
      * destruct (IH rest eq_refl index use equations Hlookup) as [Hin H].
        split; [ right; exact Hin | exact H ].
Qed.

Lemma cut_for_joins_has (cut_after : Scope.t) :
  ∀ (envs : list (Index.t * Typing_env.t)) (joined : Joined_envs.t),
    Join_env.cut_for_joins cut_after envs = Some joined ->
    NoDup (map fst envs) ->
    ∀ (index : Index.t) (use : Typing_env.t),
      (index, use) ∈ envs ->
      Typing_env.is_bottom use = false ->
      ∃ level : Typing_env_level.t,
        Typing_env.cut use cut_after = Some level
        ∧ Keyed.lookup Index.equal index joined
          = Some (use, Typing_env_level.equations level).
Proof.
  induction envs as [ | [i tenv] envs IH];
    intros joined Hcut Hdistinct index use Hin Hbottom; [ destruct Hin | ].
  cbn [Join_env.cut_for_joins] in Hcut.
  destruct (Join_env.cut_for_joins cut_after envs) as [rest | ] eqn:Hrest;
    [ | discriminate ].
  cbn [map fst] in Hdistinct.
  inversion Hdistinct as [ | ? ? Hfresh Hdistinct0]; subst.
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. rewrite Hbottom in Hcut.
    destruct (Typing_env.cut use cut_after) as [level | ]; [ | discriminate ].
    injection Hcut as <-. exists level. split; [ reflexivity | ].
    rewrite keyed_lookup_cons.
    rewrite (proj2 (index_equal_decides index index) eq_refl). reflexivity.
  - destruct (IH rest eq_refl Hdistinct0 index use Hin Hbottom)
      as [level [Hlevel Hlookup]].
    exists level. split; [ exact Hlevel | ].
    assert (Hne : Index.equal index i = false).
    { destruct (Index.equal index i) eqn:Heq; [ | reflexivity ].
      exfalso. apply index_equal_decides in Heq. subst i. apply Hfresh.
      apply in_map_iff. exists (index, use). split; [ reflexivity | exact Hin ].
    }
    destruct (Typing_env.is_bottom tenv).
    + injection Hcut as <-. exact Hlookup.
    + destruct (Typing_env.cut tenv cut_after) as [level0 | ];
        [ | discriminate ].
      injection Hcut as <-. rewrite keyed_lookup_cons, Hne. exact Hlookup.
Qed.

Lemma get_nth_lookup (envs : Joined_envs.t) (index : Index.t)
    (use : Typing_env.t) :
  Joined_envs.get_nth_joined_env envs index = Some use
  <-> ∃ equations : Keyed.finite_map name flambda_type,
        Keyed.lookup Index.equal index envs = Some (use, equations).
Proof.
  unfold Joined_envs.get_nth_joined_env.
  destruct (Keyed.lookup Index.equal index envs) as [[use' equations] | ];
    cbn [option_map fst]; split.
  - intros H. injection H as ->. exists equations. reflexivity.
  - intros [equations' H]. injection H as -> _. reflexivity.
  - discriminate.
  - intros [equations' H]. discriminate.
Qed.

(* ---------------------------------------------------------------------- *)
(* The joined environments and their levels through the phases             *)
(* ---------------------------------------------------------------------- *)

Lemma join_aliases_same_envs (p : Join_env.parameters) (fuel : nat)
    (envs : Joined_envs.t) (exists0 : variable -> bool) :
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
    same_envs s s'.
Proof.
  induction names as [ | n names IH]; intros s s' Hfold.
  - injection Hfold as <-. reflexivity.
  - cbn [List.fold_left] in Hfold.
    destruct n as [x | sym]; [ | exact (IH s s' Hfold) ].
    destruct (exists0 x); [ | exact (IH s s' Hfold) ].
    destruct (Join_env.import_or_delay_n_way_join_type p fuel s x
                (Join_env.types_of_name_in_levels envs (Name_var x)))
      as [s1 | ] eqn:Hs1.
    + exact (same_envs_trans _ _ _
               (import_or_delay_same_envs _ _ _ _ _ _ Hs1) (IH s1 s' Hfold)).
    + exfalso. clear -Hfold.
      induction names as [ | n' names IH]; [ discriminate | exact (IH Hfold) ].
Qed.

Section Rounds_envs.

Context (jt : Join_env.n_way_join_type) (choices : Join_env.choices).

Hypothesis jt_keeps : join_of_types_keeps_envs jt.

Lemma round_same_envs (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations) :
  Join_env.n_way_join_round choices jt s equations relations
  = Some (relations', s') ->
  same_envs s s'.
Proof.
  intros Hround. unfold Join_env.n_way_join_round in Hround.
  refine (fold_left_invariant _
            (fun a : Join_env.inverse_relations * Join_env.t =>
               same_envs s (snd a))
            (fun b => eq_refl) _ _ (relations, s) (relations', s') eq_refl
            Hround).
  intros [relations0 s0] [relations1 s1] [var types] _ H0 Hstep.
  cbn [fst snd] in Hstep, H0 |- *.
  destruct (jt s0 (Index.bindings types)) as [[joined s2] | ] eqn:Hjoin;
    [ | discriminate ].
  pose proof (same_envs_trans _ _ _ H0 (jt_keeps _ _ _ _ Hjoin)) as H2.
  destruct joined as [ty | ].
  - destruct (Join_env.recover_relations_on choices s2 var types).
    + destruct (Join_env.recover_inverse_relations relations0 (Name_var var)
                  ty) as [[ty' relations2] | ]; [ | discriminate ].
      destruct (Join_env.add_equation_in_target_env s2 var ty') as [s3 | ]
        eqn:Hadd; [ | discriminate ].
      injection Hstep as _ <-.
      exact (same_envs_trans _ _ _ H2 (add_equation_same_envs _ _ _ _ Hadd)).
    + destruct (Join_env.add_equation_in_target_env s2 var ty) as [s3 | ]
        eqn:Hadd; [ | discriminate ].
      injection Hstep as _ <-.
      exact (same_envs_trans _ _ _ H2 (add_equation_same_envs _ _ _ _ Hadd)).
  - injection Hstep as _ <-. exact H2.
Qed.

Lemma delayed_equations_same_envs :
  ∀ (fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type)),
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    same_envs s s'.
Proof.
  induction fuel as [ | fuel IH];
    intros s s' relations relations' equations Hloop; [ discriminate | ].
  cbn [Join_env.n_way_join_delayed_equations] in Hloop.
  destruct (Keyed.is_empty equations).
  - injection Hloop as _ <-. reflexivity.
  - destruct (Join_env.n_way_join_round choices jt s equations relations)
      as [[relations1 s1] | ] eqn:Hround; [ | discriminate ].
    exact (same_envs_trans _ _ _ (round_same_envs _ _ _ _ _ Hround)
             (IH s1 s' relations1 relations' _ Hloop)).
Qed.

Lemma loop_same_envs (p : Join_env.parameters) :
  ∀ (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    same_envs s s'.
Proof.
  induction depth as [ | depth IH];
    intros fuel s s' relations relations' existentials Hloop;
    cbn [Join_env.n_way_join_loop] in Hloop.
  - injection Hloop as _ <-. reflexivity.
  - destruct (Keyed.is_empty existentials).
    + injection Hloop as _ <-. reflexivity.
    + destruct (Join_env.equations_of_existentials p s (compact existentials))
        as [equations | ]; [ | discriminate ].
      destruct (Join_env.n_way_join_delayed_equations fuel choices jt s
                  relations equations) as [[relations1 s1] | ] eqn:Hdelayed;
        [ | discriminate ].
      exact (same_envs_trans _ _ _
               (delayed_equations_same_envs _ _ _ _ _ _ Hdelayed)
               (IH fuel s1 s' relations1 relations' _ Hloop)).
Qed.

End Rounds_envs.

(* ---------------------------------------------------------------------- *)
(* The definitions at the normal name mode                                 *)
(* ---------------------------------------------------------------------- *)

Lemma simples_at_normal_mode_find (envs : Joined_envs.t) :
  ∀ (simples : list (Index.t * simple))
    (at_normal_mode : Simples_in_joined_envs.t),
    Analysis.simples_at_normal_mode envs simples = Some (Some at_normal_mode)
    ->
    ∀ (index : Index.t) (canonical : simple),
      Simples_in_joined_envs.find index at_normal_mode = Some canonical ->
      ∃ (simple : simple) (use : Typing_env.t),
        (index, simple) ∈ simples
        ∧ Joined_envs.get_nth_joined_env envs index = Some use
        ∧ Typing_env.get_canonical_simple_exn use (Some Name_mode.normal) None
            simple
          = Some (Some canonical).
Proof.
  induction simples as [ | [index0 simple0] simples IH];
    intros at_normal_mode H index canonical Hfind;
    cbn [Analysis.simples_at_normal_mode] in H.
  - injection H as <-. discriminate.
  - destruct (Joined_envs.get_nth_joined_env envs index0) as [use0 | ]
      eqn:Huse0; [ | discriminate ].
    destruct (Typing_env.get_canonical_simple_exn use0 (Some Name_mode.normal)
                None simple0) as [canonical0 | ] eqn:Hcanonical0;
      [ | discriminate ].
    destruct (Analysis.simples_at_normal_mode envs simples) as [rest | ]
      eqn:Hrest; [ | discriminate ].
    destruct canonical0 as [canonical0 | ]; [ | discriminate ].
    destruct rest as [rest | ]; [ | discriminate ].
    injection H as <-. unfold Simples_in_joined_envs.find in Hfind.
    rewrite keyed_lookup_cons in Hfind.
    destruct (Index.equal index index0) eqn:Heq.
    + injection Hfind as <-. apply index_equal_decides in Heq. subst index0.
      exists simple0, use0. split; [ left; reflexivity | ].
      split; [ exact Huse0 | exact Hcanonical0 ].
    + destruct (IH rest eq_refl index canonical Hfind)
        as [simple [use [Hin H]]].
      exists simple, use. split; [ right; exact Hin | exact H ].
Qed.

Lemma definitions_at_normal_mode_lookup (envs : Joined_envs.t) :
  ∀ (definitions :
       list (variable * (Simples_in_joined_envs.t * flambda_kind)))
    (result : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    Analysis.definitions_at_normal_mode envs definitions = Some result ->
    ∀ (x : variable) (at_normal_mode : Simples_in_joined_envs.t)
      (kind : flambda_kind),
      lookup x result = Some (at_normal_mode, kind) ->
      ∃ simples : Simples_in_joined_envs.t,
        (x, (simples, kind)) ∈ definitions
        ∧ Analysis.simples_at_normal_mode envs
            (Keyed.compact Index.equal simples)
          = Some (Some at_normal_mode).
Proof.
  induction definitions as [ | [var [simples kind0]] definitions IH];
    intros result H x at_normal_mode kind Hx;
    cbn [Analysis.definitions_at_normal_mode] in H.
  - injection H as <-. discriminate.
  - destruct (Analysis.simples_at_normal_mode envs
                (Keyed.compact Index.equal simples)) as [at0 | ] eqn:Hat0;
      [ | discriminate ].
    destruct (Analysis.definitions_at_normal_mode envs definitions)
      as [rest | ] eqn:Hrest; [ | discriminate ].
    injection H as <-. destruct at0 as [at0 | ].
    + rewrite lookup_cons in Hx. destruct (Pos.eqb x var) eqn:Heq.
      * injection Hx as <- <-. apply Pos.eqb_eq in Heq. subst var.
        exists simples. split; [ left; reflexivity | exact Hat0 ].
      * destruct (IH rest eq_refl x at_normal_mode kind Hx)
          as [simples' [Hin H]].
        exists simples'. split; [ right; exact Hin | exact H ].
    + destruct (IH rest eq_refl x at_normal_mode kind Hx)
        as [simples' [Hin H]].
      exists simples'. split; [ right; exact Hin | exact H ].
Qed.

(* The definitions at the normal name mode hold where the definitions
   that they are made from hold. *)
Lemma definitions_at_normal_mode_hold (envs : Joined_envs.t)
    (definitions at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (use : Typing_env.t) (statics : static_env)
    (valuation valuation' : env) :
  Analysis.definitions_at_normal_mode envs (compact definitions)
  = Some at_normal_mode ->
  Joined_envs.get_nth_joined_env envs index = Some use ->
  typing_env_satisfied_strictly statics valuation use ->
  definitions_hold definitions index valuation valuation' ->
  definitions_hold at_normal_mode index valuation valuation'.
Proof.
  intros Hnormal Huse Hsatisfied Hhold x simples' kind canonical v Hx Hfind
    Hv.
  destruct (definitions_at_normal_mode_lookup envs _ _ Hnormal x simples' kind
              Hx) as [simples [Hin Hat]].
  apply in_compact in Hin.
  destruct (simples_at_normal_mode_find envs _ _ Hat index canonical Hfind)
    as [simple [use' [Hin' [Huse' Hcanonical]]]].
  rewrite Huse in Huse'. injection Huse' as <-.
  apply (keyed_in_compact Index.equal index_equal_decides) in Hin'.
  apply (Hhold x simples kind simple v Hin Hin').
  rewrite <- (get_canonical_simple_satisfied (fun _ => False) statics valuation
                use (Some Name_mode.normal) None simple canonical Hsatisfied
                Hcanonical).
  exact Hv.
Qed.

(* ---------------------------------------------------------------------- *)
(* The three functions keep the joined environments and their levels       *)
(* ---------------------------------------------------------------------- *)

Theorem import_type_keeps_envs : import_type_keeps_envs_statement.
Proof.
  intros p fuel s s' ty Himport.
  exact (import_vars_same_envs _ (free_variables ty)
           (import_var_same_envs p fuel) s s' Himport).
Qed.

Theorem n_way_join_simples_keeps_envs
  : n_way_join_simples_keeps_envs_statement.
Proof.
  intros p fuel s s' kind simples result Hjoin.
  unfold Join_env.n_way_join_simples in Hjoin.
  destruct simples as [ | first rest].
  { injection Hjoin as _ <-. reflexivity. }
  destruct (Joined_envs.get_canonical_simples_ignoring_name_mode _ _)
    as [canonicals | ]; [ | discriminate ].
  destruct (Join_env.n_way_join_canonical_simples s canonicals)
    as [[simple0 | var | canonicals'] | ]; [ | | | discriminate ].
  - injection Hjoin as _ <-. reflexivity.
  - destruct (Join_env.import_var_transitive p fuel s var) as [s1 | ]
      eqn:Himport; [ | discriminate ].
    injection Hjoin as _ <-.
    exact (import_var_same_envs p fuel _ _ _ Himport).
  - destruct (Join_env.existential_for_these_canonicals p s canonicals' kind)
      as [x s1] eqn:Hmake.
    injection Hjoin as _ <-.
    exact (existential_same_envs _ _ _ _ _ _ Hmake).
Qed.

Theorem n_way_join_env_extension_keeps_envs
  : n_way_join_env_extension_keeps_envs_statement.
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
    + injection Hfold as <-. reflexivity.
    + cbn [List.fold_left fst snd] in Hfold.
      destruct (Join_env.import_name_transitive p fuel s n) as [s2 | ]
        eqn:Hname.
      * assert (Hfirst : same_envs s s2).
        { destruct n as [x | sym].
          - exact (import_var_same_envs p fuel _ _ _ Hname).
          - injection Hname as <-. reflexivity. }
        destruct (Join_env.import_type_transitive p fuel s2 ty) as [s3 | ]
          eqn:Htype.
        -- exact (same_envs_trans _ _ _ Hfirst
                    (same_envs_trans _ _ _
                       (import_type_keeps_envs p fuel _ _ _ Htype)
                       (IH _ Hfold))).
        -- exfalso. clear -Hfold.
           induction equations as [ | e equations IH];
             [ discriminate | exact (IH Hfold) ].
      * exfalso. clear -Hfold.
        induction equations as [ | e equations IH];
          [ discriminate | exact (IH Hfold) ].
  - injection Hjoin as _ <-. reflexivity.
Qed.

Theorem n_way_join_keeps_envs : n_way_join_keeps_envs_statement.
Proof.
  intros p fuel_of_env fuel expand_head reduce_head_of_kind_naked_immediate
    type_eqb block_shape_eqb s s' ts joined Hstep.
  refine
    (n_way_join_keeps Join_env.t
       (fun s0 s0' => Join_env.joined_envs s0' = Join_env.joined_envs s0)
       Join_env.joined_env (Join_env.n_way_join_simples p fuel_of_env)
       (Join_env.import_type p fuel_of_env)
       (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel
       (fun s0 => eq_refl) same_envs_trans _ _ _ s s' ts joined Hstep).
  - intros s0 s0' kind simples result Hsimples.
    exact (n_way_join_simples_keeps_envs p fuel_of_env s0 s0' kind simples
             result Hsimples).
  - intros s0 s0' ty Himport.
    exact (import_type_keeps_envs p fuel_of_env s0 s0' ty Himport).
  - intros s0 s0' extensions joined0 Hextension.
    exact (n_way_join_env_extension_keeps_envs p fuel_of_env s0 s0'
             extensions joined0 Hextension).
Qed.

(* ---------------------------------------------------------------------- *)
(* The join that Simplify calls, up to what it adds to the target          *)
(* ---------------------------------------------------------------------- *)

Theorem join_with_true : join_with_true_statement.
Proof.
  intros A meet_env meet_expanded_head ops p fuel choices jt mh avoid inputs
    target analysis Hjt Hkeeps Hfresh Heqb Hconst Hinputs Hkinded Hagree
    [index1 [use1 [Hat1 Hbottom1]]] Hjoin.
  destruct Hinputs
    as [[_ Huses_wf] [_ [_ [_ [Havoid_source Havoid_uses]]]]].
  unfold join_with, cut_and_n_way_join_with_analysis in Hjoin.
  cbv zeta in Hjoin.
  destruct (Join_env.cut_for_joins _ _) as [envs | ] eqn:Hcut;
    [ | discriminate ].
  change (Join_env.cut_for_joins (cut_after inputs) (numbered_uses inputs)
          = Some envs) in Hcut.
  destruct (Join_env.cut_and_n_way_join0 _ _ _ _ _ _ _ _ _ _)
    as [[[menv_t definitions] names] | ] eqn:Hjoin0; [ | discriminate ].
  destruct (Analysis.create _ envs definitions) as [analysis' | ]
    eqn:Hanalysis; [ | discriminate ].
  injection Hjoin as <- <-.
  assert (Hfound :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              use_at inputs index use ∧ Typing_env.is_bottom use = false).
  { intros index use Huse. apply get_nth_lookup in Huse.
    destruct Huse as [equations Hlookup].
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup) as [Hin [Hb _]].
    split; [ apply numbered_uses_in; exact Hin | exact Hb ]. }
  assert (Hhas :
            ∀ (index : Index.t) (use : Typing_env.t),
              use_at inputs index use ->
              Typing_env.is_bottom use = false ->
              ∃ level : Typing_env_level.t,
                Typing_env.cut use (cut_after inputs) = Some level
                ∧ Keyed.lookup Index.equal index envs
                  = Some (use, Typing_env_level.equations level)).
  { intros index use Hat Hb.
    exact (cut_for_joins_has _ _ _ Hcut (numbered_uses_distinct inputs) index
             use (proj2 (numbered_uses_in inputs index use) Hat) Hb). }
  assert (Hempty : Keyed.is_empty envs = false).
  { destruct (Keyed.is_empty envs) eqn:He; [ | reflexivity ].
    destruct (Hhas index1 use1 Hat1 Hbottom1) as [level [_ Hlookup]].
    rewrite (proj1 (keyed_is_empty_lookup Index.t Index.equal
                      index_equal_decides _ envs) He index1) in Hlookup.
    discriminate. }
  assert (Hsource : ∀ x : variable, knows (source inputs) x -> x ∈ avoid).
  { intros x Hx. apply among_In. exact (Havoid_source x Hx). }
  assert (Huses :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              knows use x ->
              x ∈ avoid).
  { intros index use x Huse Hx. apply among_In.
    exact (Havoid_uses index use x (proj1 (Hfound index use Huse)) Hx). }
  assert (Hinvariant :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_invariant use).
  { intros index use Huse.
    exact (Huses_wf index use (proj1 (Hfound index use Huse))). }
  assert (Hkinded_envs :
            ∀ (index : Index.t) (use : Typing_env.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              typing_env_kinded use).
  { intros index use Huse.
    exact (Hkinded index use (proj1 (Hfound index use Huse))). }
  assert (Hlevels : levels_found envs).
  { intros index use equations n ty Hlookup Hn.
    destruct (cut_for_joins_found _ _ _ Hcut _ _ _ Hlookup)
      as [Hin [_ [level [Hlevel ->]]]].
    apply numbered_uses_in in Hin.
    exact (cut_equations_found use (cut_after inputs) level n ty
             (proj2 (Huses_wf index use Hin)) Hlevel Hn). }
  assert (Hkinds :
            ∀ (index : Index.t) (use : Typing_env.t) (x : variable)
              (kind : flambda_kind) (ty : flambda_type)
              (binding_time_and_mode : Binding_time.With_name_mode.t),
              Joined_envs.get_nth_joined_env envs index = Some use ->
              simple_of_kind use (Var x) kind ->
              Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types (source inputs))
              = Some (ty, binding_time_and_mode) ->
              FlambdaTypeGrammar.kind ty = kind).
  { intros index use x kind ty btm Huse Hk Hx.
    assert (Hsrc :
              simple_of_kind (source inputs) (Var x)
                (FlambdaTypeGrammar.kind ty)).
    { apply (simple_of_kind_name (source inputs) (Name_var x)). left.
      exists ty, btm. split; [ exact Hx | reflexivity ]. }
    destruct Hsrc as [ty0 Hfind0].
    destruct (Hagree index use (Name_var x) (FlambdaTypeGrammar.kind ty) ty0
                (proj1 (Hfound index use Huse))
                (simple_of_kind_var_known use x kind Hk) Hfind0)
      as [ty' Hfind'].
    exact (simple_of_kind_unique use (Var x) (FlambdaTypeGrammar.kind ty) kind
             (ex_intro _ ty' Hfind') Hk). }
  unfold Join_env.cut_and_n_way_join0 in Hjoin0. rewrite Hempty in Hjoin0.
  cbv zeta in Hjoin0.
  set (s0 :=
         Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid (source inputs)))
    in *.
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
  injection Hjoin0 as <- <- _.
  destruct (first_phases_true (source inputs) jt choices p fuel avoid envs s1
              s2 s3 relations2 relations3 Hjt Hfresh Heqb Hconst Hsource Huses
              Hinvariant Hkinded_envs Hlevels Hkinds Haliases Hdelayed Hloop)
    as [Htrue3 [Hrelations3 Hgrows3]].
  pose proof (first_phases_kinded (source inputs) jt choices p fuel avoid envs
                s1 s2 s3 relations2 relations3 Hjt Hfresh Heqb Hconst Hsource
                Huses Hinvariant Hkinded_envs Hlevels Hkinds Haliases Hdelayed
                Hloop) as Hstate3.
  assert (Hnonempty1 : relations_nonempty []).
  { intros r names' Hr. discriminate Hr. }
  pose proof (loop_relations_nonempty _ _ _ _ _ _ _ _ _ _
                (delayed_equations_relations_nonempty _ _ _ _ _ _ _ _
                   Hnonempty1 Hdelayed)
                Hloop) as Hnonempty3.
  assert (Henvs3 : Join_env.joined_envs s3 = envs).
  { unfold Join_env.join_aliases_into_bindings in Haliases.
    pose proof (join_aliases_same_envs p fuel envs _ _ s0 s1 Haliases) as H1.
    pose proof (delayed_equations_same_envs jt choices Hkeeps _ _ _ _ _ _
                  Hdelayed) as H2.
    pose proof (loop_same_envs jt choices Hkeeps p _ _ _ _ _ _ _ Hloop) as H3.
    rewrite H3, H2, H1. reflexivity. }
  assert (Huse3 :
            ∀ (index : Index.t) (use : Typing_env.t),
              Join_env.joined_env s3 index = Some use
              <-> use_at inputs index use ∧ Typing_env.is_bottom use = false).
  { intros index use. unfold Join_env.joined_env. rewrite Henvs3. split.
    - exact (Hfound index use).
    - intros [Hat Hb]. destruct (Hhas index use Hat Hb) as [level [_ Hl]].
      apply get_nth_lookup. exists (Typing_env_level.equations level).
      exact Hl. }
  assert (Hkinded3 : joined_envs_are typing_env_kinded s3).
  { intros index use Huse.
    exact (Hkinded index use (proj1 (proj1 (Huse3 index use) Huse))). }
  destruct (define_variables_true meet_env meet_expanded_head ops fuel mh
              (source inputs) s3 _ target' relations3 definitions' Htrue3
              Hstate3 Hkinded3 Hrelations3 Hnonempty3 Hdefine)
    as [kept [equations [relation_equations [menv' [Hkept [Hadd1 [Hadd2
          Hruns]]]]]]].
  unfold join_adds_true.
  exists s3, relations3, kept, equations, relation_equations, menv', target'.
  split; [ exact Htrue3 | ]. split; [ exact Hstate3 | ].
  split; [ exact Hrelations3 | ]. split; [ exact Huse3 | ].
  split; [ exact Hkept | ]. split; [ exact Hadd1 | ].
  split; [ exact Hadd2 | ]. split; [ reflexivity | ].
  intros index use statics valuation Hat Hb Hsatisfied Htags Hjoined
    Himported Hscrutinees.
  pose proof (proj2 (Huse3 index use) (conj Hat Hb)) as Huse.
  pose proof (proj2 (proj2 Htrue3)) as Hsource3.
  assert (Htags3 : tags_at s3 use statics valuation).
  { intros simple v tag fields Hscope Heval Hblock.
    apply (Htags simple v tag fields); [ | exact Heval | exact Hblock ].
    intros x Hx. pose proof (Hscope x Hx) as Hk. unfold known_to in Hk.
    apply orb_true_iff in Hk. destruct Hk as [Hk | Hk].
    - right. exists use. split; [ exact Hat | exact Hk ].
    - left. rewrite <- Hsource3. exact Hk. }
  assert (Hjoined3 : joined_have_values s3 valuation).
  { intros x Hk Hevery. apply (Hjoined x).
    - rewrite <- Hsource3. exact Hk.
    - intros index' use' Hat' Hb'.
      destruct (Hhas index' use' Hat' Hb') as [level [Hlevel Hlookup]].
      rewrite <- Henvs3 in Hlookup.
      pose proof (Hevery index' use' _ Hlookup) as Hbound.
      apply keyed_is_bound_lookup in Hbound. destruct Hbound as [ty Hty].
      exists level, ty. split; [ exact Hlevel | exact Hty ]. }
  pose proof (aliases_have_values_of_values (source inputs) s3 index use
                valuation Htrue3 Huse Himported Hjoined3) as Haliases3.
  destruct (Hruns index use statics valuation Huse Hsatisfied Htags3 Haliases3
              Hscrutinees) as [Hequations [Hrelation_equations Hdefinitions]].
  unfold Analysis.create in Hanalysis.
  destruct (Analysis.definitions_at_normal_mode envs (compact definitions'))
    as [at_normal_mode | ] eqn:Hnormal; [ | discriminate ].
  injection Hanalysis as <-.
  cbn [Analysis.definitions_in_joined_envs
         Analysis.canonical_definitions_at_normal_mode].
  split; [ exact Hequations | ]. split; [ exact Hrelation_equations | ].
  split; [ exact Hdefinitions | ].
  apply (definitions_at_normal_mode_hold envs definitions' at_normal_mode
           index use statics valuation _ Hnormal); [ | exact Hsatisfied
                                                     | exact Hdefinitions ].
  unfold Join_env.joined_env in Huse. rewrite Henvs3 in Huse. exact Huse.
Qed.

Theorem compiler_join_true : compiler_join_true_statement.
Proof.
  intros A meet_env meet_expanded_head ops p fuel fuel_of_env fuel_of_types
    expand_head reduce mh avoid inputs target analysis Hfresh Heqb Hextension
    Hconst Hexpand Hreduce Hinputs Hkinded Hagree Hsome Hjoin.
  exact (join_with_true A meet_env meet_expanded_head ops p fuel
           Join_env.compiler_choices
           (n_way_join_of_types p fuel_of_env expand_head reduce
              fuel_of_types)
           mh avoid inputs target analysis
           (n_way_join_of_types_true (source inputs) p fuel_of_env expand_head
              reduce fuel_of_types Hfresh Heqb Hextension Hexpand Hreduce)
           (n_way_join_keeps_envs p fuel_of_env fuel_of_types expand_head
              reduce type_eqb block_shape_eqb)
           Hfresh Heqb Hconst Hinputs Hkinded Hagree Hsome Hjoin).
Qed.

(* ---------------------------------------------------------------------- *)
(* The valuation of the target is in range                                 *)
(* ---------------------------------------------------------------------- *)

Lemma in_tabulate {A : Type} (f : positive -> option A)
    (names : list positive) (x : positive) (a : A) :
  (x, a) ∈ tabulate f names -> f x = Some a.
Proof.
  unfold tabulate. intros Hin. apply in_flat_map in Hin.
  destruct Hin as [name [_ Hin]]. destruct (f name) as [a' | ] eqn:Hf;
    [ | destruct Hin ].
  destruct Hin as [Heq | []]. injection Heq as <- <-. exact Hf.
Qed.

Theorem target_valuation_in_range : target_valuation_in_range_statement.
Proof.
  intros s index use valuation Hdefinitions Hrange x v Hin.
  unfold target_valuation in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - apply in_tabulate in Hin. unfold value_of_definition in Hin.
    destruct (lookup x (Join_env.definitions_of_existentials s))
      as [[simples kind] | ] eqn:Hdefinition; [ | discriminate ].
    destruct (Simples_in_joined_envs.find index simples) as [simple | ]
      eqn:Hfind; [ | discriminate ].
    destruct simple as [y | sym | c]; cbn in Hin.
    + exact (Hrange y v (lookup_in y v valuation Hin)).
    + injection Hin as <-. reflexivity.
    + injection Hin as <-.
      exact (Hdefinitions x simples kind index c Hdefinition Hfind).
  - unfold filter_bindings in Hin. apply in_tabulate in Hin.
    destruct (lookup x valuation) as [v' | ] eqn:Hlookup; [ | discriminate ].
    destruct (known_to s use x); [ | discriminate ].
    injection Hin as <-. exact (Hrange x v' (lookup_in x v' valuation Hlookup)).
Qed.
