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



(* Proofs of the statements of FlambdaProversSpec.v. Nothing here needs
   review.

   Each prover on heads is proved once ([generic_value_sound]), and two
   lemmas take that through [Provers.gen_value_to_proof] and
   [Provers.gen_value_to_meet]. What a prover reads of an expanded type
   is through [expanded_value_cases] and the lemmas beside it, which
   are all that is used of the soundness of [expand_head].

   The provers that read no relation of a type of naked immediates are
   proved once for two notions of satisfaction of the environment: the
   strict one, of FlambdaProversSpec.v, and the one that exempts the
   names that hold poison, of FlambdaProversPoisonSpec.v (the section
   "The two notions of satisfaction"). For such a prover the lemma
   whose name ends in [_in] is for both, and the lemma of the same
   name without it, which follows it, is its instance at the strict
   notion; the theorems here are of the strict notion, and those of
   FlambdaProversPoisonProofs.v of the other. The provers that read
   relations are proved for the strict notion only. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaValueProofs
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeMeaningBoolSpec
  FlambdaTypeMeaningBoolProofs FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaTypeAccessSpec FlambdaTypeAccessProofs FlambdaBindingTime
  FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaExpandHead FlambdaExpandHeadSpec
  FlambdaExpandHeadProofs FlambdaProvers FlambdaProversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The view of an expanded type                                             *)
(* ======================================================================== *)

Definition descr_of_oub {head : Type} (oub : or_unknown_or_bottom head)
    : type_descr head :=
  match oub with
  | Unknown => Unknown
  | Bottom => Bottom
  | Ok head => Ok (No_alias head)
  end.

(* The type that a view of an expanded type is a view of. *)
Definition type_of_descr_oub (d : descr_oub) : flambda_type :=
  match d with
  | Oub_value oub => Type_value (descr_of_oub oub)
  | Oub_naked_immediate oub => Type_naked_immediate (descr_of_oub oub)
  | Oub_naked_number k oub => Type_naked_number k (descr_of_oub oub)
  | Oub_rec_info oub => Type_rec_info (descr_of_oub oub)
  | Oub_region oub => Type_region (descr_of_oub oub)
  end.

Lemma type_of_descr_oub_of_expanded (expanded : Expanded_type.t) :
  type_of_descr_oub (descr_oub_of_expanded expanded)
  = Expanded_type.to_type expanded.
Proof.
  destruct expanded as [k [ | head | ]];
    unfold descr_oub_of_expanded, Expanded_type.to_type; simpl.
  - destruct k as [ | [ | | | | | | | | | | | ] | | ]; reflexivity.
  - destruct head; reflexivity.
  - destruct k as [ | [ | | | | | | | | | | | ] | | ]; reflexivity.
Qed.

(* ======================================================================== *)
(* Expanding keeps well-formedness                                          *)
(* ======================================================================== *)

Lemma of_non_alias_type_well_formed (ty : flambda_type)
    (expanded : Expanded_type.t) :
  type_well_formed ty = true ->
  Expanded_type.of_non_alias_type ty = Some expanded ->
  type_well_formed (Expanded_type.to_type expanded) = true.
Proof.
  intros Hwf Hexp.
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; simpl in Hexp;
    try discriminate Hexp; injection Hexp as <-;
    first
      [ exact Hwf
      | apply unknown_well_formed
      | apply bottom_well_formed ].
Qed.

Lemma create_const_well_formed (c : const) :
  type_well_formed (Expanded_type.to_type (Expanded_type.create_const c))
  = true.
Proof.
  destruct c; try reflexivity. apply unknown_well_formed.
Qed.

Lemma expand_head_of_alias_type_well_formed (tenv : Typing_env.t)
    (kind : flambda_kind) (s : simple) (expanded : Expanded_type.t) :
  types_well_formed tenv ->
  expand_head_of_alias_type tenv kind s = Some expanded ->
  type_well_formed (Expanded_type.to_type expanded) = true.
Proof.
  intros Htypes.
  assert (Hname :
    ∀ n : name,
      (let* ty := Typing_env.find tenv n (Some kind) in
       Expanded_type.of_non_alias_type ty)
      = Some expanded ->
      type_well_formed (Expanded_type.to_type expanded) = true).
  { intros n.
    destruct (Typing_env.find tenv n (Some kind)) as [ty | ] eqn:Hfind;
      [ | discriminate ].
    apply of_non_alias_type_well_formed.
    exact (find_well_formed _ _ _ _ Htypes Hfind). }
  unfold expand_head_of_alias_type.
  destruct s as [x | sym | c].
  - apply Hname.
  - apply Hname.
  - intros [= <-]. apply create_const_well_formed.
Qed.

Lemma expand_head_well_formed (tenv : Typing_env.t) (t : flambda_type)
    (expanded : Expanded_type.t) :
  well_formed_input tenv t ->
  expand_head tenv t = Some expanded ->
  type_well_formed (Expanded_type.to_type expanded) = true.
Proof.
  intros [Htypes Hwf]. unfold expand_head.
  destruct (get_alias_opt t) as [s | ].
  - destruct
      (Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types)
         None s)
      as [[canonical | ] | ]; [ | | discriminate ].
    + apply expand_head_of_alias_type_well_formed. exact Htypes.
    + apply of_non_alias_type_well_formed. apply unknown_well_formed.
  - apply of_non_alias_type_well_formed. exact Hwf.
Qed.

Lemma provers_expand_head_well_formed (tenv : Typing_env.t)
    (t : flambda_type) (d : descr_oub) :
  well_formed_input tenv t ->
  Provers.expand_head tenv t = Some d ->
  type_well_formed (type_of_descr_oub d) = true.
Proof.
  intros Hinput. unfold Provers.expand_head.
  destruct (expand_head tenv t) as [expanded | ] eqn:Hexp; [ | discriminate ].
  intros [= <-]. rewrite type_of_descr_oub_of_expanded.
  exact (expand_head_well_formed _ _ _ Hinput Hexp).
Qed.

(* ======================================================================== *)
(* What a value has of the answers                                          *)
(* ======================================================================== *)

Lemma immediate_not_block (statics : static_env) (v : value) :
  is_immediate v -> block_of statics v = None.
Proof. intros [i ->]. reflexivity. Qed.

Lemma immediate_kind (v : value) : is_immediate v -> kind_of_value v = Value.
Proof. intros [i ->]. reflexivity. Qed.

Lemma block_kind (statics : static_env) (v : value) :
  is_block statics v -> kind_of_value v = Value.
Proof.
  intros [tag [fields Hblock]]. exact (block_of_kind _ _ _ _ Hblock).
Qed.

Lemma poison_not_null (v : value) : is_poison v -> v <> Value_const Null.
Proof. intros [k ->]. discriminate. Qed.

Lemma poison_no_head (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  is_poison v ->
  ~ head_of_kind_value_non_null_satisfied statics vars v head.
Proof.
  intros [k ->] Hhead.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k' contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    autorewrite with type_satisfied in Hhead.
  - destruct Hhead as [[i [Heq _]] | [tag [fields [Hblock _]]]];
      discriminate.
  - exact Hhead.
  - destruct Hhead as [c [Hboxed _]]. discriminate.
  - destruct Hhead as [slot [functions [value_slots [Hclosure _]]]].
    discriminate.
  - destruct Hhead as [bytes [Hstring _]]. discriminate.
Qed.

(* A variant with nothing known of it has every tagged immediate and
   every block. *)
Definition any_variant : head_of_kind_value_non_null :=
  Variant None Or_unknown.Unknown None Or_unknown.Unknown No_extensions false.

Lemma any_variant_satisfied (statics : static_env) (vars : env) (v : value) :
  variants statics v ->
  head_of_kind_value_non_null_satisfied statics vars v any_variant.
Proof.
  unfold any_variant. rewrite variant_satisfied.
  intros [[i ->] | [tag [fields Hblock]]].
  - left. exists i. simpl. auto.
  - right. exists tag, fields. simpl. auto.
Qed.

(* A value of a variant, as the provers read it. *)
Lemma variant_cases (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Variant is_int immediates get_tag blocks extensions is_unique) ->
  (∃ i : Z,
    v = Value_const (Tagged_immediate i)
    ∧ match immediates with
      | Or_unknown.Unknown => True
      | Or_unknown.Known ty =>
        type_satisfied statics vars (Value_const (Naked_immediate i)) ty
      end)
  ∨ (∃ (tag : Z) (fields : list value),
      block_of statics v = Some (tag, fields)
      ∧ match blocks with
        | Or_unknown.Unknown => True
        | Or_unknown.Known blocks =>
          row_like_for_blocks_satisfied statics vars tag fields blocks
        end).
Proof.
  rewrite variant_satisfied.
  intros [[i [Hv [_ [Himms _]]]]
         | [tag [fields [Hblock [_ [_ [Hblocks _]]]]]]].
  - left. exists i. auto.
  - right. exists tag, fields. auto.
Qed.

(* A head that is not a variant has no tagged immediate and no block. *)
Lemma not_variant_head (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  head_of_kind_value_non_null_satisfied statics vars v head ->
  match head with
  | Variant _ _ _ _ _ _ | Mutable_block _ => True
  | Boxed_number _ _ _ | Closures _ _ | String _ => ~ variants statics v
  end.
Proof.
  intros Hhead.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try exact I;
    intros Hvariant;
    exact
      (heads_compatible_satisfied _ _ _ _ _
         (any_variant_satisfied statics vars v Hvariant) Hhead).
Qed.

(* ======================================================================== *)
(* Sets of numbers                                                          *)
(* ======================================================================== *)

Lemma is_empty_nil {A : Type} (l : list A) :
  Keyed.is_empty l = true -> l = [].
Proof. destruct l; [ reflexivity | discriminate ]. Qed.

Lemma in_inter (i : Z) (imms imms' : list Z) :
  i ∈ imms -> i ∈ imms' -> i ∈ Keyed.inter Z.eqb imms imms'.
Proof.
  intros Hin Hin'.
  apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
  rewrite (keyed_among_inter Z Z.eqb z_eqb_decides_equality).
  rewrite (proj2 (keyed_among_in Z Z.eqb z_eqb_decides_equality i imms) Hin).
  rewrite
    (proj2 (keyed_among_in Z Z.eqb z_eqb_decides_equality i imms') Hin').
  reflexivity.
Qed.

Lemma get_singleton_in (i imm : Z) (imms : list Z) :
  Keyed.get_singleton Z.eqb imms = Some imm -> i ∈ imms -> i = imm.
Proof.
  intros Hsingleton Hin.
  pose proof
    (proj1 (keyed_get_singleton Z Z.eqb z_eqb_decides_equality imm imms)
       Hsingleton i) as Hamong.
  rewrite (proj2 (keyed_among_in Z Z.eqb z_eqb_decides_equality i imms) Hin)
    in Hamong.
  symmetry in Hamong. exact (proj1 (Z.eqb_eq _ _) Hamong).
Qed.

(* ======================================================================== *)
(* The two notions of satisfaction                                          *)
(* ======================================================================== *)

(* The proofs of the provers that read no relation are made once for
   the two notions of satisfaction of a typing environment: the strict
   one, and the one that exempts the names that hold poison. A lemma
   whose name ends in [_in] is for both; the lemma of the same name
   without it is the one for the strict notion. *)
Inductive mode : Type :=
  | Strict
  | Exempt.

Definition sat (m : mode) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) : Prop :=
  match m with
  | Strict => typing_env_satisfied_strictly statics valuation tenv
  | Exempt => typing_env_satisfied statics valuation tenv
  end.

(* The values that a prover is sound of: any, strictly; where poison is
   exempt, those that are not poison. *)
Definition ok (m : mode) (v : value) : Prop :=
  match m with
  | Strict => True
  | Exempt => ~ is_poison v
  end.

(* What is asked of the view of an expanded type, for a value of the
   type to have it: nothing, strictly; where poison is exempt, that it
   is not a head of naked immediates with a relation on a name that
   holds poison. A name whose type is such a head is exempt. *)
Definition scrutinees_ok (m : mode) (valuation : env) (d : descr_oub)
    : Prop :=
  match m with
  | Strict => True
  | Exempt =>
    match d with
    | Oub_naked_immediate (Ok head) =>
      ∀ (relation : Relation.t) (scrutinee : name),
        (relation, scrutinee) ∈ inverse_relations head ->
        ~ holds_poison valuation scrutinee
    | _ => True
    end
  end.

Lemma sat_but (m : mode) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) :
  sat m statics valuation tenv ->
  ∃ exempt : name -> Prop,
    typing_env_satisfied_but exempt statics valuation tenv.
Proof. destruct m; intros Hsat; eexists; exact Hsat. Qed.

Lemma ok_not_poison (m : mode) (v : value) : ~ is_poison v -> ok m v.
Proof. destruct m; intros H; [ exact I | exact H ]. Qed.

Lemma ok_naked_immediate (m : mode) (i : Z) :
  ok m (Value_const (Naked_immediate i)).
Proof. apply ok_not_poison. intros [k Hk]. discriminate Hk. Qed.

Lemma scrutinees_ok_value (m : mode) (valuation : env)
    (oub : or_unknown_or_bottom head_of_kind_value) :
  scrutinees_ok m valuation (Oub_value oub).
Proof. destruct m; exact I. Qed.

Lemma scrutinees_ok_naked_number (m : mode) (valuation : env)
    (k : naked_number_kind) (oub : or_unknown_or_bottom (list Z)) :
  scrutinees_ok m valuation (Oub_naked_number k oub).
Proof. destruct m; exact I. Qed.

(* [expand_head_sound] of FlambdaExpandHeadProofs.v, for both notions:
   where poison is exempt, of a value that is not poison, and of a type
   that does not expand to a relation on a name that holds poison. *)
Lemma expand_head_sound_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (ty : flambda_type)
    (expanded : Expanded_type.t) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  scrutinees_ok m valuation (descr_oub_of_expanded expanded) ->
  type_satisfied statics valuation v ty ->
  expand_head tenv ty = Some expanded ->
  type_satisfied statics valuation v (Expanded_type.to_type expanded).
Proof.
  destruct m; intros Hsat Hok Hscrutinees Hty Hexpand.
  - exact (expand_head_sound _ _ _ _ _ _ Hsat Hty Hexpand).
  - cbn [sat ok] in Hsat, Hok.
    destruct (expand_head_cases tenv ty expanded Hexpand)
      as [[_ Hexpanded]
         | [s [Halias [[_ Hexpanded]
                      | [[c [Hcanonical ->]]
                        | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
    + exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded Hty).
    + apply (of_non_alias_type_sound _ _ _ _ _ Hexpanded).
      apply (proj2 (unknown_satisfied _ _ _ _)).
      exact (proj2 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hty)).
    + apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
      destruct Hty as [Hvalue _].
      rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsat
                    Hcanonical) in Hvalue.
      injection Hvalue as <-. apply create_const_sound.
    + apply (of_non_alias_type_sound _ _ _ _ _ Hexpanded).
      apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
      destruct Hty as [Hvalue _].
      rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsat
                    Hcanonical) in Hvalue.
      apply (find_not_poison _ _ _ _ _ _ _ Hsat Hfind Hvalue Hok).
      intros [head [relation [scrutinee [-> [Hin [_ Hpoison]]]]]].
      injection Hexpanded as <-.
      exact (Hscrutinees relation scrutinee Hin Hpoison).
Qed.

(* The kind of a value of a type is that of the type expanded, whatever
   is exempt: what is asked of the value of a name fixes its kind. *)
Lemma expand_head_kind_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (ty : flambda_type)
    (expanded : Expanded_type.t) (v : value) :
  sat m statics valuation tenv ->
  type_satisfied statics valuation v ty ->
  expand_head tenv ty = Some expanded ->
  kind_of_value v = kind (Expanded_type.to_type expanded).
Proof.
  intros Hsat Hty Hexpand.
  destruct (sat_but m _ _ _ Hsat) as [exempt Hbut].
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - apply (type_satisfied_kind statics valuation).
    exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded Hty).
  - apply (type_satisfied_kind statics valuation).
    apply (of_non_alias_type_sound _ _ _ _ _ Hexpanded).
    apply (proj2 (unknown_satisfied _ _ _ _)).
    exact (proj2 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hty)).
  - apply (type_satisfied_kind statics valuation).
    apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
    destruct Hty as [Hvalue _].
    rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hbut
                  Hcanonical) in Hvalue.
    injection Hvalue as <-. apply create_const_sound.
  - apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
    destruct Hty as [Hvalue _].
    rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hbut
                  Hcanonical) in Hvalue.
    rewrite (proj2 (of_non_alias_type_kind _ _ Hexpanded)).
    exact
      (has_type_unless_kind _ _ _ _ _
         (find_satisfied _ _ _ _ _ _ _ _ Hbut Hfind Hvalue)).
Qed.

(* ======================================================================== *)
(* The provers, by the soundness of [expand_head]                           *)
(* ======================================================================== *)

Lemma expand_head_descr_sound_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type)
    (d : descr_oub) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  scrutinees_ok m valuation d ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some d ->
  type_satisfied statics valuation v (type_of_descr_oub d).
Proof.
  intros Hsat Hok Hscrutinees Hv. unfold Provers.expand_head.
  destruct (expand_head tenv t) as [expanded | ] eqn:Hexp; [ | discriminate ].
  intros [= Hd]. subst d. rewrite type_of_descr_oub_of_expanded.
  exact (expand_head_sound_in _ _ _ _ _ _ Hsat Hok Hscrutinees Hv Hexp).
Qed.

Lemma expand_head_descr_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (d : descr_oub) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some d ->
  type_satisfied statics valuation v (type_of_descr_oub d).
Proof.
  intros Hsat.
  exact
    (expand_head_descr_sound_in (m := Strict) statics valuation tenv t d v
       Hsat I I).
Qed.

(* The kind of a value of a type is that of the view of the type
   expanded. *)
Lemma expanded_kind_in {m : mode} (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (d : descr_oub) (v : value) :
  sat m statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some d ->
  kind_of_value v = kind (type_of_descr_oub d).
Proof.
  intros Hsat Hv. unfold Provers.expand_head.
  destruct (expand_head tenv t) as [expanded | ] eqn:Hexp; [ | discriminate ].
  intros [= Hd]. subst d. rewrite type_of_descr_oub_of_expanded.
  exact (expand_head_kind_in _ _ _ _ _ _ Hsat Hv Hexp).
Qed.

Lemma expanded_kind (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (d : descr_oub) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some d ->
  kind_of_value v = kind (type_of_descr_oub d).
Proof. exact (expanded_kind_in (m := Strict) statics valuation tenv t d v). Qed.

(* What a value of a type has, when the type expanded is of kind
   value. *)
Lemma expanded_value_cases_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type)
    (oub : or_unknown_or_bottom head_of_kind_value) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_value oub) ->
  match oub with
  | Unknown => kind_of_value v = Value
  | Bottom => False
  | Ok (Head_of_kind_value non_null is_null) =>
    (v = Value_const Null ∧ is_null <> Not_null)
    ∨ (v <> Value_const Null
       ∧ match non_null with
         | Unknown => kind_of_value v = Value
         | Bottom => False
         | Ok head =>
           head_of_kind_value_non_null_satisfied statics valuation v head
         end)
  end.
Proof.
  intros Hsat Hok Hv Hd.
  pose proof
    (expand_head_descr_sound_in _ _ _ _ _ _ Hsat Hok
       (scrutinees_ok_value _ _ _) Hv Hd)
    as Hexp.
  simpl in Hexp. rewrite type_satisfied_value in Hexp.
  destruct oub as [ | [non_null is_null] | ]; simpl in Hexp.
  - exact Hexp.
  - rewrite head_of_kind_value_satisfied_eq in Hexp.
    destruct Hexp as [[Hnull His_null] | [Hnot_null [Hnon_null _]]].
    + left. split; [ exact Hnull | ].
      intros ->. exact His_null.
    + right. split; [ exact Hnot_null | exact Hnon_null ].
  - exact Hexp.
Qed.

Lemma expanded_value_cases (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type)
    (oub : or_unknown_or_bottom head_of_kind_value) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_value oub) ->
  match oub with
  | Unknown => kind_of_value v = Value
  | Bottom => False
  | Ok (Head_of_kind_value non_null is_null) =>
    (v = Value_const Null ∧ is_null <> Not_null)
    ∨ (v <> Value_const Null
       ∧ match non_null with
         | Unknown => kind_of_value v = Value
         | Bottom => False
         | Ok head =>
           head_of_kind_value_non_null_satisfied statics valuation v head
         end)
  end.
Proof.
  intros Hsat.
  exact
    (expanded_value_cases_in (m := Strict) statics valuation tenv t oub v Hsat
       I).
Qed.

(* The same when it is of kind naked immediate. *)
Lemma expanded_naked_immediate_cases_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type)
    (oub : or_unknown_or_bottom head_of_kind_naked_immediate) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  scrutinees_ok m valuation (Oub_naked_immediate oub) ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_naked_immediate oub) ->
  match oub with
  | Unknown => True
  | Bottom => False
  | Ok head => head_of_kind_naked_immediate_satisfied statics valuation v head
  end.
Proof.
  intros Hsat Hok Hscrutinees Hv Hd.
  pose proof
    (expand_head_descr_sound_in _ _ _ _ _ _ Hsat Hok Hscrutinees Hv Hd)
    as Hexp.
  simpl in Hexp. rewrite type_satisfied_naked_immediate in Hexp.
  destruct oub as [ | head | ]; simpl in Hexp.
  - exact I.
  - exact Hexp.
  - exact Hexp.
Qed.

Lemma expanded_naked_immediate_cases (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type)
    (oub : or_unknown_or_bottom head_of_kind_naked_immediate) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_naked_immediate oub) ->
  match oub with
  | Unknown => True
  | Bottom => False
  | Ok head => head_of_kind_naked_immediate_satisfied statics valuation v head
  end.
Proof.
  intros Hsat.
  exact
    (expanded_naked_immediate_cases_in (m := Strict) statics valuation tenv t
       oub v Hsat I I).
Qed.

(* A type that expands to bottom has no value: where poison is exempt,
   none that is not poison. *)
Lemma is_bottom_sound_in {m : mode} (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  Provers.is_bottom tenv t = Some true ->
  ~ type_satisfied statics valuation v t.
Proof.
  intros Hsat Hok Hbottom Hv. unfold Provers.is_bottom in Hbottom.
  destruct (expand_head tenv t) as [expanded | ] eqn:Hexp;
    [ | discriminate ].
  injection Hbottom as Hbottom.
  assert
    (Hscrutinees : scrutinees_ok m valuation (descr_oub_of_expanded expanded)).
  { unfold Expanded_type.is_bottom in Hbottom.
    destruct expanded as [k [ | head | ]]; try discriminate Hbottom.
    destruct m; [ exact I | ]. destruct k as [ | [] | | ]; exact I. }
  pose proof (expand_head_sound_in _ _ _ _ _ _ Hsat Hok Hscrutinees Hv Hexp)
    as Hexpanded.
  unfold Expanded_type.is_bottom in Hbottom.
  unfold Expanded_type.to_type in Hexpanded.
  destruct (Expanded_type.descr expanded); try discriminate Hbottom.
  exact (bottom_satisfied _ _ _ _ Hexpanded).
Qed.

Lemma is_bottom_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  Provers.is_bottom tenv t = Some true ->
  ~ type_satisfied statics valuation v t.
Proof.
  intros Hsat.
  exact (is_bottom_sound_in (m := Strict) statics valuation tenv t v Hsat I).
Qed.

(* ======================================================================== *)
(* A prover on heads, and the three ways to make a prover of it             *)
(* ======================================================================== *)

(* No condition on the head that a prover on heads is given, and the
   one that [well_formed_input] gives. *)
Definition any_head (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) : Prop :=
  True.

Definition well_formed_head (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) : Prop :=
  types_well_formed tenv
  ∧ head_of_kind_value_non_null_well_formed head = true.

(* A prover on heads is sound: [meet_shortcut_sound], for a head that
   is not null in place of a type. *)
Definition generic_value_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    input tenv head ->
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    domain statics v ->
    match prove_gen tenv head with
    | Some (Generic_proof.Proved x) => holds statics valuation x v
    | Some Generic_proof.Invalid => False
    | Some Generic_proof.Unknown | None => True
    end.

(* The same for both notions of satisfaction; [generic_value_sound] is
   the one for the strict notion. *)
Definition generic_value_sound_in (m : mode) {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) (v : value),
    sat m statics valuation tenv ->
    input tenv head ->
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    domain statics v ->
    match prove_gen tenv head with
    | Some (Generic_proof.Proved x) => holds statics valuation x v
    | Some Generic_proof.Invalid => False
    | Some Generic_proof.Unknown | None => True
    end.

(* The two forms of FlambdaProversSpec.v for both notions of
   satisfaction, of the values that a prover is sound of. *)
Definition prover_sound_in (m : mode) {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    sat m statics valuation tenv ->
    ok m v ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    prover tenv t = Some (Proof_of_property.Proved x) ->
    holds statics valuation x v.

Definition meet_shortcut_sound_in (m : mode) {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    sat m statics valuation tenv ->
    ok m v ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    domain statics v ->
    match prover tenv t with
    | Some (Meet_shortcut.Known_result x) => holds statics valuation x v
    | Some Meet_shortcut.Invalid => False
    | Some Meet_shortcut.Need_meet | None => True
    end.

Lemma prover_sound_strictly {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  prover_sound_in Strict prover input holds ->
  prover_sound prover input holds.
Proof.
  intros H statics valuation tenv t x v Hsat.
  exact (H statics valuation tenv t x v Hsat I).
Qed.

Arguments prover_sound_strictly {A prover input holds} _.

Lemma meet_shortcut_sound_strictly {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  meet_shortcut_sound_in Strict prover input domain holds ->
  meet_shortcut_sound prover input domain holds.
Proof.
  intros H statics valuation tenv t v Hsat.
  exact (H statics valuation tenv t v Hsat I).
Qed.

Arguments meet_shortcut_sound_strictly {A prover input domain holds} _.

(* The lemma for the strict notion, from the one for both. *)
Ltac strictly lemma :=
  first
    [ exact lemma
    | exact (prover_sound_strictly lemma)
    | exact (meet_shortcut_sound_strictly lemma) ].

(* The condition on a type gives the condition on the head that the
   type expands to. *)
Definition input_to_head
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    : Prop :=
  ∀ (tenv : Typing_env.t) (t : flambda_type)
    (head : head_of_kind_value_non_null) (is_null : is_null),
    input tenv t ->
    Provers.expand_head tenv t
    = Some (Oub_value (Ok (Head_of_kind_value (Ok head) is_null))) ->
    input_head tenv head.

Lemma any_input_to_head : input_to_head any_input any_head.
Proof. intros tenv t head is_null _ _. exact I. Qed.

Lemma well_formed_input_to_head :
  input_to_head well_formed_input well_formed_head.
Proof.
  intros tenv t head is_null Hinput Hd.
  split; [ exact (proj1 Hinput) | ].
  exact (provers_expand_head_well_formed _ _ _ Hinput Hd).
Qed.

Lemma gen_value_to_proof_sound_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  generic_value_sound_in m prove_gen input_head (fun _ _ => True) holds ->
  prover_sound_in m (Provers.gen_value_to_proof prove_gen) input holds.
Proof.
  intros Hinput Hgen statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.gen_value_to_proof in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
  specialize
    (Hgen statics valuation tenv head v Hsat (Hinput _ _ _ _ Hin Hd) Hhead I).
  destruct (prove_gen tenv head) as [[y | | ] | ];
    try discriminate Hproved.
  injection Hproved as ->. exact Hgen.
Qed.

Lemma gen_value_to_proof_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  generic_value_sound prove_gen input_head (fun _ _ => True) holds ->
  prover_sound (Provers.gen_value_to_proof prove_gen) input holds.
Proof.
  intros Hinput Hgen.
  exact
    (prover_sound_strictly
       (gen_value_to_proof_sound_in (m := Strict) prove_gen input input_head
          holds Hinput Hgen)).
Qed.

Lemma gen_value_to_meet_sound_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (domain domain' : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  (∀ (statics : static_env) (v : value),
    domain' statics v ->
    kind_of_value v = Value ∧ v <> Value_const Null ∧ domain statics v) ->
  generic_value_sound_in m prove_gen input_head domain holds ->
  meet_shortcut_sound_in m (Provers.gen_value_to_meet prove_gen) input domain'
    holds.
Proof.
  intros Hinput Hdomain Hgen statics valuation tenv t v Hsat Hok Hin Hv Hdom.
  destruct (Hdomain _ _ Hdom) as [Hkind [Hnot_null Hdom']].
  unfold Provers.gen_value_to_meet.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind_in _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  rewrite Hkind in Hk.
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hk.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  destruct oub as [ | [[ | head | ] is_null] | ]; simpl in Hcases |- *.
  - exact I.
  - exact I.
  - destruct Hcases as [[Hnull _] | [_ Hhead]]; [ contradiction | ].
    specialize
      (Hgen statics valuation tenv head v Hsat (Hinput _ _ _ _ Hin Hd) Hhead
         Hdom').
    destruct (prove_gen tenv head) as [[y | | ] | ]; exact Hgen.
  - destruct Hcases as [[Hnull _] | [_ []]]. contradiction.
  - exact Hcases.
Qed.

Lemma gen_value_to_meet_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (domain domain' : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  (∀ (statics : static_env) (v : value),
    domain' statics v ->
    kind_of_value v = Value ∧ v <> Value_const Null ∧ domain statics v) ->
  generic_value_sound prove_gen input_head domain holds ->
  meet_shortcut_sound (Provers.gen_value_to_meet prove_gen) input domain'
    holds.
Proof.
  intros Hinput Hdomain Hgen.
  exact
    (meet_shortcut_sound_strictly
       (gen_value_to_meet_sound_in (m := Strict) prove_gen input input_head
          domain domain' holds Hinput Hdomain Hgen)).
Qed.

(* [Provers.gen_value_to_gen], of a value of kind value. *)
Lemma gen_value_to_gen_sound_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value) :
  generic_value_sound_in m prove_gen any_head domain holds ->
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  kind_of_value v = Value ->
  domain statics v ->
  match Provers.gen_value_to_gen prove_gen tenv t with
  | Some (Generic_proof.Proved x) => holds statics valuation x v
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hgen Hsat Hok Hv Hkind Hdom.
  unfold Provers.gen_value_to_gen.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind_in _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  rewrite Hkind in Hk.
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hk.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  destruct oub as [ | [[ | head | ] [ | is_null]] | ];
    simpl in Hcases |- *; try exact I.
  - destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
    specialize (Hgen statics valuation tenv head v Hsat I Hhead Hdom).
    destruct (prove_gen tenv head) as [[y | | ] | ]; exact Hgen.
  - destruct Hcases as [[_ Hnot] | [_ []]]. contradiction.
  - exact Hcases.
Qed.

Lemma gen_value_to_gen_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value) :
  generic_value_sound prove_gen any_head domain holds ->
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  kind_of_value v = Value ->
  domain statics v ->
  match Provers.gen_value_to_gen prove_gen tenv t with
  | Some (Generic_proof.Proved x) => holds statics valuation x v
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hgen Hsat.
  exact
    (gen_value_to_gen_sound_in (m := Strict) prove_gen domain holds statics
       valuation tenv t v Hgen Hsat I).
Qed.

(* A prover made with [Provers.gen_value_to_proof] or
   [Provers.gen_value_to_meet] says nothing of a value that is poison,
   whatever the prover on heads. *)
Lemma gen_value_to_proof_silent {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A)) :
  prover_silent_on_poison (Provers.gen_value_to_proof prove_gen).
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hproved.
  unfold Provers.gen_value_to_proof in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
  exact (poison_no_head _ _ _ _ Hpoison Hhead).
Qed.

Lemma gen_value_to_meet_silent {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A)) :
  meet_shortcut_silent_on_poison (Provers.gen_value_to_meet prove_gen).
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hknown.
  unfold Provers.gen_value_to_meet in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] is_null] | ] | oub | k oub | oub | oub];
    try discriminate Hknown.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[Hnull _] | [_ Hhead]].
  - exact (poison_not_null _ Hpoison Hnull).
  - exact (poison_no_head _ _ _ _ Hpoison Hhead).
Qed.

(* The answer [Known_result] of a prover made with
   [Provers.gen_value_to_meet] holds of every value of the type that
   is not null, if the answer [Proved] of the prover on heads holds of
   every value of the head. *)
Lemma gen_value_to_meet_known_sound_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (holds : static_env -> env -> A -> value -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value) :
  (∀ (head : head_of_kind_value_non_null),
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    holds statics valuation x v) ->
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  v <> Value_const Null ->
  Provers.gen_value_to_meet prove_gen tenv t
  = Some (Meet_shortcut.Known_result x) ->
  holds statics valuation x v.
Proof.
  intros Hgen Hsat Hok Hv Hnot_null Hknown.
  unfold Provers.gen_value_to_meet in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] is_null] | ] | oub | k oub | oub | oub];
    try discriminate Hknown.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[Hnull _] | [_ Hhead]]; [ contradiction | ].
  destruct (prove_gen tenv head) as [[y | | ] | ] eqn:Hproved;
    try discriminate Hknown.
  injection Hknown as ->. exact (Hgen head Hhead Hproved).
Qed.

Lemma gen_value_to_meet_known_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (holds : static_env -> env -> A -> value -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value) :
  (∀ (head : head_of_kind_value_non_null),
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    holds statics valuation x v) ->
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  v <> Value_const Null ->
  Provers.gen_value_to_meet prove_gen tenv t
  = Some (Meet_shortcut.Known_result x) ->
  holds statics valuation x v.
Proof.
  intros Hgen Hsat.
  exact
    (gen_value_to_meet_known_sound_in (m := Strict) prove_gen holds statics
       valuation tenv t x v Hgen Hsat I).
Qed.

(* ======================================================================== *)
(* The scrutinee questions                                                  *)
(* ======================================================================== *)

(* A variant whose immediates have no value is a block, and one whose
   blocks are bottom is a tagged immediate. *)
Lemma variant_no_immediates_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (v : value)
    (is_int : option variable)
    (imms : flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  sat m statics valuation tenv ->
  head_of_kind_value_non_null_satisfied statics valuation v
    (Variant is_int (Or_unknown.Known imms) get_tag blocks extensions
       is_unique) ->
  Provers.is_bottom tenv imms = Some true ->
  ~ is_immediate v
  ∧ ∃ (tag : Z) (fields : list value),
      block_of statics v = Some (tag, fields)
      ∧ match blocks with
        | Or_unknown.Unknown => True
        | Or_unknown.Known blocks =>
          row_like_for_blocks_satisfied statics valuation tag fields blocks
        end.
Proof.
  intros Hsat Hhead Hbottom.
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
    as [[i [-> Himms]] | [tag [fields [Hblock Hblocks]]]].
  - destruct
      (is_bottom_sound_in _ _ _ _ _ Hsat (ok_naked_immediate _ _) Hbottom
         Himms).
  - split; [ | exists tag, fields; split; assumption ].
    intros Himmediate.
    rewrite (immediate_not_block statics v Himmediate) in Hblock.
    discriminate Hblock.
Qed.

Lemma variant_no_immediates (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (v : value) (is_int : option variable)
    (imms : flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  typing_env_satisfied_strictly statics valuation tenv ->
  head_of_kind_value_non_null_satisfied statics valuation v
    (Variant is_int (Or_unknown.Known imms) get_tag blocks extensions
       is_unique) ->
  Provers.is_bottom tenv imms = Some true ->
  ~ is_immediate v
  ∧ ∃ (tag : Z) (fields : list value),
      block_of statics v = Some (tag, fields)
      ∧ match blocks with
        | Or_unknown.Unknown => True
        | Or_unknown.Known blocks =>
          row_like_for_blocks_satisfied statics valuation tag fields blocks
        end.
Proof.
  strictly
    (variant_no_immediates_in (m := Strict) statics valuation tenv v is_int
       imms get_tag blocks extensions is_unique).
Qed.

Lemma variant_no_blocks (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Variant is_int immediates get_tag (Or_unknown.Known blocks) extensions
       is_unique) ->
  row_like_for_blocks_is_bottom blocks = true ->
  ∃ i : Z,
    v = Value_const (Tagged_immediate i)
    ∧ match immediates with
      | Or_unknown.Unknown => True
      | Or_unknown.Known ty =>
        type_satisfied statics vars (Value_const (Naked_immediate i)) ty
      end.
Proof.
  intros Hhead Hbottom.
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
    as [Himmediate | [tag [fields [_ Hblocks]]]].
  - exact Himmediate.
  - destruct
      (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom Hblocks).
Qed.

Lemma prove_is_int_generic_value_sound_in {m : mode} (variant_only : bool) :
  generic_value_sound_in m (Provers.prove_is_int_generic_value variant_only)
    any_head
    (fun statics v => variant_only = true -> variants statics v)
    (fun _ _ => is_int_is).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead Hdomain.
  pose proof (not_variant_head _ _ _ _ Hhead) as Hnot_variant.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl.
  - destruct blocks as [blocks | ], immediates as [imms | ].
    + destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks;
        destruct (Provers.is_bottom tenv imms) as [[ | ] | ] eqn:Himms;
        simpl; try exact I.
      * destruct (variant_no_immediates_in _ _ _ _ _ _ _ _ _ _ Hsat Hhead Himms)
          as [Hnot _].
        destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks)
          as [i [-> _]].
        apply Hnot. exists i. reflexivity.
      * destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks)
          as [i [-> _]].
        exists i. reflexivity.
      * exact
          (proj1
             (variant_no_immediates_in _ _ _ _ _ _ _ _ _ _ Hsat Hhead Himms)).
    + destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks;
        simpl; try exact I.
      destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks)
        as [i [-> _]].
      exists i. reflexivity.
    + destruct (Provers.is_bottom tenv imms) as [[ | ] | ] eqn:Himms;
        simpl; try exact I.
      exact
        (proj1 (variant_no_immediates_in _ _ _ _ _ _ _ _ _ _ Hsat Hhead Himms)).
    + exact I.
  - rewrite mutable_block_head_satisfied in Hhead. destruct Hhead.
  - destruct variant_only; simpl.
    + exact (Hnot_variant (Hdomain eq_refl)).
    + intros Himmediate. apply Hnot_variant. left. exact Himmediate.
  - destruct variant_only; simpl.
    + exact (Hnot_variant (Hdomain eq_refl)).
    + intros Himmediate. apply Hnot_variant. left. exact Himmediate.
  - destruct variant_only; simpl.
    + exact (Hnot_variant (Hdomain eq_refl)).
    + intros Himmediate. apply Hnot_variant. left. exact Himmediate.
Qed.

Lemma prove_is_int_generic_value_sound (variant_only : bool) :
  generic_value_sound (Provers.prove_is_int_generic_value variant_only)
    any_head
    (fun statics v => variant_only = true -> variants statics v)
    (fun _ _ => is_int_is).
Proof.
  strictly (prove_is_int_generic_value_sound_in (m := Strict) variant_only).
Qed.

Lemma prove_is_int_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_is_int any_input (fun _ _ => is_int_is).
Proof.
  apply (gen_value_to_proof_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - intros statics valuation tenv head v Hsat Hin Hhead _.
    apply
      (prove_is_int_generic_value_sound_in false _ _ _ _ _ Hsat Hin Hhead).
    discriminate.
Qed.

Lemma prove_is_int_sound_proof :
  prover_sound Provers.prove_is_int any_input (fun _ _ => is_int_is).
Proof.
  strictly (prove_is_int_sound_proof_in (m := Strict)).
Qed.

Lemma meet_is_int_variant_only_sound_proof_in {m : mode} :
  meet_shortcut_sound_in m Provers.meet_is_int_variant_only any_input variants
    (fun _ _ => is_int_is).
Proof.
  apply
    (gen_value_to_meet_sound_in _ any_input any_head
       (fun statics v => true = true -> variants statics v)).
  - exact any_input_to_head.
  - intros statics v Hvariant.
    split; [ | split; [ | intros _; exact Hvariant ] ].
    + destruct Hvariant as [Himmediate | Hblock].
      * exact (immediate_kind _ Himmediate).
      * exact (block_kind _ _ Hblock).
    + destruct Hvariant as [[i ->] | [tag [fields Hblock]]].
      * discriminate.
      * intros ->. discriminate Hblock.
  - exact (prove_is_int_generic_value_sound_in true).
Qed.

Lemma meet_is_int_variant_only_sound_proof :
  meet_shortcut_sound Provers.meet_is_int_variant_only any_input variants
    (fun _ _ => is_int_is).
Proof.
  strictly (meet_is_int_variant_only_sound_proof_in (m := Strict)).
Qed.

Lemma prove_is_not_a_pointer_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_is_not_a_pointer any_input
    (fun _ _ => is_not_a_pointer_is).
Proof.
  intros statics valuation tenv t x v Hsat Hok _ Hv Hproved.
  unfold Provers.prove_is_not_a_pointer,
    Provers.prove_is_not_a_pointer_generic_value in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved;
    pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases;
    simpl in Hcases.
  - (* A head that is not null. *)
    destruct Hcases as [[_ Hnot] | [Hnot_null Hhead]]; [ contradiction | ].
    pose proof
      (prove_is_int_generic_value_sound_in false _ _ _ _ _ Hsat I Hhead
         (fun H => False_ind _ (diff_false_true H)))
      as Hint.
    destruct (Provers.prove_is_int_generic_value false tenv head)
      as [[[ | ] | | ] | ];
      try discriminate Hproved; injection Hproved as <-; simpl in Hint |- *.
    + left. exact Hint.
    + split; [ exact Hint | exact Hnot_null ].
  - (* A head that may be null. *)
    destruct Hcases as [[Hnull _] | [Hnot_null Hhead]].
    + destruct (Provers.prove_is_int_generic_value false tenv head)
        as [[[ | ] | | ] | ];
        try discriminate Hproved; injection Hproved as <-.
      right. exact Hnull.
    + pose proof
        (prove_is_int_generic_value_sound_in false _ _ _ _ _ Hsat I Hhead
           (fun H => False_ind _ (diff_false_true H)))
        as Hint.
      destruct (Provers.prove_is_int_generic_value false tenv head)
        as [[[ | ] | | ] | ];
        try discriminate Hproved; injection Hproved as <-; simpl in Hint.
      left. exact Hint.
  - (* Null and nothing else. *)
    injection Hproved as <-.
    destruct Hcases as [[Hnull _] | [_ []]]. right. exact Hnull.
Qed.

Lemma prove_is_not_a_pointer_sound_proof :
  prover_sound Provers.prove_is_not_a_pointer any_input
    (fun _ _ => is_not_a_pointer_is).
Proof.
  strictly (prove_is_not_a_pointer_sound_proof_in (m := Strict)).
Qed.

Lemma prove_is_not_a_pointer_silent :
  prover_silent_on_poison Provers.prove_is_not_a_pointer.
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hproved.
  unfold Provers.prove_is_not_a_pointer,
    Provers.prove_is_not_a_pointer_generic_value in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved;
    pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd) as Hcases;
    simpl in Hcases.
  - destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
    exact (poison_no_head _ _ _ _ Hpoison Hhead).
  - destruct Hcases as [[Hnull _] | [_ Hhead]].
    + exact (poison_not_null _ Hpoison Hnull).
    + exact (poison_no_head _ _ _ _ Hpoison Hhead).
  - destruct Hcases as [[Hnull _] | [_ []]].
    exact (poison_not_null _ Hpoison Hnull).
Qed.

Lemma prove_is_null_generic_sound_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  kind_of_value v = Value ->
  match Provers.prove_is_null_generic tenv t with
  | Some (Generic_proof.Proved x) => is_null_is x v
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hsat Hok Hv Hkind. unfold Provers.prove_is_null_generic.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind_in _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  rewrite Hkind in Hk.
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hk.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  destruct oub as [ | [[ | head | ] [ | is_null]] | ];
    simpl in Hcases |- *; try exact I.
  - destruct Hcases as [[_ Hnot] | [Hnot_null _]];
      [ contradiction | exact Hnot_null ].
  - destruct Hcases as [[_ Hnot] | [Hnot_null _]];
      [ contradiction | exact Hnot_null ].
  - destruct Hcases as [[_ Hnot] | [_ []]]. contradiction.
  - destruct Hcases as [[Hnull _] | [_ []]]. exact Hnull.
  - exact Hcases.
Qed.

Lemma prove_is_null_generic_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  kind_of_value v = Value ->
  match Provers.prove_is_null_generic tenv t with
  | Some (Generic_proof.Proved x) => is_null_is x v
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hsat.
  exact
    (prove_is_null_generic_sound_in (m := Strict) statics valuation tenv t v
       Hsat I).
Qed.

Lemma meet_is_null_sound_proof_in {m : mode} :
  meet_shortcut_sound_in m Provers.meet_is_null any_input (values_of_kind Value)
    (fun _ _ => is_null_is).
Proof.
  intros statics valuation tenv t v Hsat Hok _ Hv Hkind.
  pose proof (prove_is_null_generic_sound_in _ _ _ _ _ Hsat Hok Hv Hkind)
    as Hnull.
  unfold Provers.meet_is_null.
  destruct (Provers.prove_is_null_generic tenv t) as [[x | | ] | ];
    exact Hnull.
Qed.

Lemma meet_is_null_sound_proof :
  meet_shortcut_sound Provers.meet_is_null any_input (values_of_kind Value)
    (fun _ _ => is_null_is).
Proof.
  strictly (meet_is_null_sound_proof_in (m := Strict)).
Qed.

Lemma meet_is_null_poison_proof (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  is_poison v ->
  Provers.meet_is_null tenv t <> Some (Meet_shortcut.Known_result true).
Proof.
  intros Hsat Hv Hpoison Hknown.
  unfold Provers.meet_is_null, Provers.prove_is_null_generic in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hknown.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[Hnull _] | [_ []]].
  exact (poison_not_null _ Hpoison Hnull).
Qed.

(* ======================================================================== *)
(* Tags                                                                     *)
(* ======================================================================== *)

Lemma prove_get_tag_generic_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_get_tag_generic_value any_head
    (fun _ _ => True) (fun statics _ => block_with_tag_among statics).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead _.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I.
  destruct immediates as [imms | ]; [ | exact I ].
  destruct (Provers.is_bottom tenv imms) as [[ | ] | ] eqn:Himms;
    simpl; try exact I.
  destruct (variant_no_immediates_in _ _ _ _ _ _ _ _ _ _ Hsat Hhead Himms)
    as [_ [tag [fields [Hblock Hblocks]]]].
  destruct blocks as [blocks | ]; [ | exact I ].
  destruct (row_like_for_blocks_all_tags blocks) as [tags | ] eqn:Htags;
    [ | exact I ].
  pose proof (all_tags_satisfied _ _ _ _ _ _ Htags Hblocks) as Hin.
  destruct (Keyed.is_empty tags) eqn:Hempty.
  - rewrite (is_empty_nil _ Hempty) in Hin. destruct Hin.
  - exists tag, fields. split; assumption.
Qed.

Lemma prove_get_tag_generic_value_sound :
  generic_value_sound Provers.prove_get_tag_generic_value any_head
    (fun _ _ => True) (fun statics _ => block_with_tag_among statics).
Proof.
  strictly (prove_get_tag_generic_value_sound_in (m := Strict)).
Qed.

Lemma prove_get_tag_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_get_tag any_input
    (fun statics _ => block_with_tag_among statics).
Proof.
  apply (gen_value_to_proof_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_get_tag_generic_value_sound_in.
Qed.

Lemma prove_get_tag_sound_proof :
  prover_sound Provers.prove_get_tag any_input
    (fun statics _ => block_with_tag_among statics).
Proof.
  strictly (prove_get_tag_sound_proof_in (m := Strict)).
Qed.

(* ======================================================================== *)
(* Naked immediates                                                         *)
(* ======================================================================== *)

(* What is known of a number admits the number [i]. *)
Definition proof_admits (i : Z) (proof : Generic_proof.t (list Z)) : Prop :=
  match proof with
  | Generic_proof.Proved imms => i ∈ imms
  | Generic_proof.Unknown => True
  | Generic_proof.Invalid => False
  end.

Definition answer_admits (i : Z)
    (answer : option (Generic_proof.t (list Z))) : Prop :=
  match answer with
  | Some proof => proof_admits i proof
  | None => True
  end.

Lemma these_immediates_admits (i : Z) (imms : list Z) :
  i ∈ imms -> proof_admits i (Provers.these_immediates imms).
Proof.
  intros Hin. unfold Provers.these_immediates.
  destruct imms as [ | imm imms]; [ destruct Hin | exact Hin ].
Qed.

Lemma meet_these_immediates_admits (i : Z) (imms : list Z)
    (proof : Generic_proof.t (list Z)) :
  i ∈ imms ->
  proof_admits i proof ->
  proof_admits i (Provers.meet_these_immediates imms proof).
Proof.
  intros Hin Hproof.
  destruct proof as [imms' | | ]; simpl.
  - apply these_immediates_admits. exact (in_inter _ _ _ Hin Hproof).
  - apply these_immediates_admits. exact Hin.
  - exact Hproof.
Qed.

Lemma meet_this_immediate_admits (i : Z) (proof : Generic_proof.t (list Z)) :
  proof_admits i proof ->
  proof_admits i (Provers.meet_this_immediate i proof).
Proof.
  apply meet_these_immediates_admits. left. reflexivity.
Qed.

(* One relation of the head of a naked immediate: the name has a value
   that the relation relates to the number. *)
Lemma prove_naked_immediates_relation_admits (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (relation : Relation.t)
    (n : name) (i : Z) (imms : Generic_proof.t (list Z)) :
  typing_env_satisfied_strictly statics valuation tenv ->
  (∃ w : value,
    value_of_name valuation n = Some w
    ∧ relation_holds statics relation w i) ->
  proof_admits i imms ->
  answer_admits i
    (Provers.prove_naked_immediates_relation tenv relation n imms).
Proof.
  intros Hsat [w [Hw Hrelation]] Himms.
  unfold Provers.prove_naked_immediates_relation.
  assert (Hkind : kind_of_value w = Value).
  { destruct relation; simpl in Hrelation.
    - destruct Hrelation as [[-> _] | [Hkind _]];
        [ reflexivity | exact Hkind ].
    - destruct Hrelation as [[Himmediate _] | [Hblock _]].
      + exact (immediate_kind _ Himmediate).
      + exact (block_kind _ _ Hblock).
    - destruct Hrelation as [Himmediate | [[fields Hblock] _]].
      + exact (immediate_kind _ Himmediate).
      + exact (block_of_kind _ _ _ _ Hblock). }
  assert (Hty :
    type_satisfied statics valuation w
      (alias_type_of Value (simple_of_name n))).
  { apply alias_satisfied. split; [ exact Hw | exact Hkind ]. }
  destruct relation; simpl in Hrelation.
  - (* [Is_null] *)
    pose proof (prove_is_null_generic_sound _ _ _ _ _ Hsat Hty Hkind)
      as Hnull.
    destruct
      (Provers.prove_is_null_generic tenv
         (alias_type_of Value (simple_of_name n)))
      as [[b | | ] | ];
      simpl; try exact I; try exact Himms; try exact Hnull.
    assert (Hi : i = Z.b2z b).
    { destruct b; simpl in Hnull |- *.
      - destruct Hrelation as [[_ ->] | [_ [Hnot _]]];
          [ reflexivity | contradiction ].
      - destruct Hrelation as [[Hw_null _] | [_ [_ ->]]];
          [ contradiction | reflexivity ]. }
    rewrite <- Hi. exact (meet_this_immediate_admits _ _ Himms).
  - (* [Is_int] *)
    assert (Hvariant : variants statics w).
    { destruct Hrelation as [[Himmediate _] | [Hblock _]];
        [ left | right ]; assumption. }
    pose proof
      (gen_value_to_gen_sound _ _ _ _ _ _ _ _
         (prove_is_int_generic_value_sound true) Hsat Hty Hkind
         (fun _ => Hvariant))
      as Hint.
    unfold Provers.prove_is_int_generic.
    destruct
      (Provers.gen_value_to_gen (Provers.prove_is_int_generic_value true)
         tenv (alias_type_of Value (simple_of_name n)))
      as [[b | | ] | ];
      simpl; try exact I; try exact Himms; try exact Hint.
    assert (Hi : i = Z.b2z b).
    { destruct b; simpl in Hint |- *.
      - destruct Hrelation as [[_ ->] | [[tag [fields Hblock]] _]];
          [ reflexivity | ].
        rewrite (immediate_not_block statics w Hint) in Hblock.
        discriminate Hblock.
      - destruct Hrelation as [[Himmediate _] | [_ ->]];
          [ contradiction | reflexivity ]. }
    rewrite <- Hi. exact (meet_this_immediate_admits _ _ Himms).
  - (* [Get_tag] *)
    pose proof
      (gen_value_to_gen_sound _ _ _ _ _ _ _ _
         prove_get_tag_generic_value_sound Hsat Hty Hkind I)
      as Htag.
    unfold Provers.prove_get_tag_generic.
    destruct
      (Provers.gen_value_to_gen Provers.prove_get_tag_generic_value tenv
         (alias_type_of Value (simple_of_name n)))
      as [[tags | | ] | ];
      simpl; try exact I; try exact Himms; try exact Htag.
    apply meet_these_immediates_admits; [ | exact Himms ].
    destruct Htag as [tag [fields [Hblock Hin]]].
    destruct Hrelation as [Himmediate | [[fields' Hblock'] _]].
    + rewrite (immediate_not_block statics w Himmediate) in Hblock.
      discriminate Hblock.
    + rewrite Hblock in Hblock'. injection Hblock' as <- _. exact Hin.
Qed.

Lemma prove_naked_immediates_fold_admits (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (i : Z)
    (relations : list (Relation.t * name)) :
  typing_env_satisfied_strictly statics valuation tenv ->
  ∀ answer : option (Generic_proof.t (list Z)),
    (∀ (relation : Relation.t) (n : name),
      (relation, n) ∈ relations ->
      ∃ w : value,
        value_of_name valuation n = Some w
        ∧ relation_holds statics relation w i) ->
    answer_admits i answer ->
    answer_admits i
      (List.fold_left
         (fun (imms : option (Generic_proof.t (list Z)))
              (relation_and_name : Relation.t * name) =>
            let* imms := imms in
            Provers.prove_naked_immediates_relation tenv
              (fst relation_and_name) (snd relation_and_name) imms)
         relations answer).
Proof.
  intros Hsat.
  induction relations as [ | [relation n] relations IH];
    intros answer Hrelations Hanswer; simpl.
  - exact Hanswer.
  - apply IH.
    + intros relation' n' Hin. apply Hrelations. right. exact Hin.
    + destruct answer as [imms | ]; [ | exact I ].
      apply (prove_naked_immediates_relation_admits statics valuation).
      * exact Hsat.
      * apply Hrelations. left. reflexivity.
      * exact Hanswer.
Qed.

Lemma prove_naked_immediates_generic_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv t ->
  type_satisfied statics valuation v t ->
  kind_of_value v = Naked_number Kind_naked_immediate ->
  match Provers.prove_naked_immediates_generic tenv t with
  | Some (Generic_proof.Proved imms) => naked_immediate_among imms v
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hsat Hinput Hv Hkind.
  unfold Provers.prove_naked_immediates_generic.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  rewrite Hkind in Hk.
  pose proof (provers_expand_head_well_formed _ _ _ Hinput Hd) as Hwf.
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hk.
  - (* A head of naked immediates. *)
    pose proof (expanded_naked_immediate_cases _ _ _ _ _ _ Hsat Hv Hd)
      as Hcases.
    destruct oub as [ | head | ]; simpl in Hcases |- *;
      [ exact I | | exact Hcases ].
    destruct Hcases as [i [-> [Hnumbers Hrelations]]].
    pose proof
      (prove_naked_immediates_fold_admits statics valuation tenv i
         (inverse_relations head) Hsat
         (Some
            match naked_immediates head with
            | Or_unknown.Known imms => Provers.these_immediates imms
            | Or_unknown.Unknown => Generic_proof.Unknown
            end)
         Hrelations)
      as Hfold.
    match type of Hfold with
    | ?P -> _ => assert (Hinitial : P)
    end.
    { simpl. destruct (naked_immediates head) as [imms | ].
      - exact (these_immediates_admits _ _ Hnumbers).
      - exact I. }
    specialize (Hfold Hinitial).
    match goal with
    | |- match ?answer with _ => _ end =>
      destruct answer as [[imms | | ] | ]
    end;
      simpl in Hfold; try exact I; try exact Hfold.
    exists i. split; [ reflexivity | exact Hfold ].
  - (* A head of naked numbers at the kind of the immediates: the type
       is not well formed. *)
    injection Hk as <-. simpl in Hwf. discriminate Hwf.
Qed.

Lemma meet_naked_immediates_sound_proof :
  meet_shortcut_sound Provers.meet_naked_immediates well_formed_input
    (values_of_kind (Naked_number Kind_naked_immediate))
    (fun _ _ => naked_immediate_among).
Proof.
  intros statics valuation tenv t v Hsat Hinput Hv Hkind.
  pose proof
    (prove_naked_immediates_generic_sound _ _ _ _ _ Hsat Hinput Hv Hkind)
    as Hsound.
  unfold Provers.meet_naked_immediates.
  destruct (Provers.prove_naked_immediates_generic tenv t)
    as [[imms | | ] | ];
    exact Hsound.
Qed.

Lemma meet_naked_immediates_silent :
  meet_shortcut_silent_on_poison Provers.meet_naked_immediates.
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hknown.
  unfold Provers.meet_naked_immediates,
    Provers.prove_naked_immediates_generic in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [oub | [ | head | ] | k oub | oub | oub];
    try discriminate Hknown.
  pose proof (expanded_naked_immediate_cases _ _ _ _ _ _ Hsat Hv Hd)
    as [i [-> _]].
  destruct Hpoison as [k Hpoison]. discriminate Hpoison.
Qed.

(* ======================================================================== *)
(* Tagged immediates                                                        *)
(* ======================================================================== *)

(* The immediates of a well-formed variant are a well-formed type. *)
Lemma well_formed_head_immediates (tenv : Typing_env.t)
    (is_int : option variable) (imms : flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  well_formed_head tenv
    (Variant is_int (Or_unknown.Known imms) get_tag blocks extensions
       is_unique) ->
  well_formed_input tenv imms.
Proof.
  intros [Htypes Hwf]. split; [ exact Htypes | ].
  rewrite variant_well_formed in Hwf.
  apply andb_prop in Hwf. destruct Hwf as [Hwf _].
  apply andb_prop in Hwf. destruct Hwf as [Hwf _].
  apply andb_prop in Hwf. exact (proj2 Hwf).
Qed.

Lemma prove_equals_tagged_immediates_value_sound :
  generic_value_sound Provers.prove_equals_tagged_immediates_value
    well_formed_head (fun _ _ => True) (fun _ _ => tagged_immediate_among).
Proof.
  intros statics valuation tenv head v Hsat Hinput Hhead _.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I.
  destruct blocks as [blocks | ]; [ | exact I ].
  destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks; [ | exact I ].
  destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks)
    as [i [-> Himms]].
  destruct immediates as [imms | ]; [ | exact I ].
  pose proof
    (prove_naked_immediates_generic_sound _ _ _ _ _ Hsat
       (well_formed_head_immediates _ _ _ _ _ _ _ Hinput) Himms eq_refl)
    as Hsound.
  destruct (Provers.prove_naked_immediates_generic tenv imms)
    as [[imms' | | ] | ];
    simpl; try exact I.
  - destruct Hsound as [i' [Heq Hin]]. injection Heq as <-.
    exists i. split; [ reflexivity | exact Hin ].
  - destruct Hsound.
Qed.

Lemma prove_equals_tagged_immediates_sound_proof :
  prover_sound Provers.prove_equals_tagged_immediates well_formed_input
    (fun _ _ => tagged_immediate_among).
Proof.
  apply (gen_value_to_proof_sound _ well_formed_input well_formed_head).
  - exact well_formed_input_to_head.
  - exact prove_equals_tagged_immediates_value_sound.
Qed.

Lemma meet_equals_tagged_immediates_value_sound :
  generic_value_sound Provers.meet_equals_tagged_immediates_value
    well_formed_head tagged_immediates (fun _ _ => tagged_immediate_among).
Proof.
  intros statics valuation tenv head v Hsat Hinput Hhead Himmediate.
  pose proof (not_variant_head _ _ _ _ Hhead) as Hnot_variant.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl.
  - destruct immediates as [imms | ]; [ | exact I ].
    destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
      as [[i [-> Himms]] | [tag [fields [Hblock _]]]].
    + pose proof
        (prove_naked_immediates_generic_sound _ _ _ _ _ Hsat
           (well_formed_head_immediates _ _ _ _ _ _ _ Hinput) Himms eq_refl)
        as Hsound.
      destruct (Provers.prove_naked_immediates_generic tenv imms)
        as [[imms' | | ] | ];
        simpl; try exact I; try exact Hsound.
      destruct Hsound as [i' [Heq Hin]]. injection Heq as <-.
      exists i. split; [ reflexivity | exact Hin ].
    + rewrite (immediate_not_block statics v Himmediate) in Hblock.
      discriminate Hblock.
  - rewrite mutable_block_head_satisfied in Hhead. exact Hhead.
  - apply Hnot_variant. left. exact Himmediate.
  - apply Hnot_variant. left. exact Himmediate.
  - apply Hnot_variant. left. exact Himmediate.
Qed.

Lemma meet_equals_tagged_immediates_sound_proof :
  meet_shortcut_sound Provers.meet_equals_tagged_immediates
    well_formed_input tagged_immediates
    (fun _ _ => tagged_immediate_among).
Proof.
  apply
    (gen_value_to_meet_sound _ well_formed_input well_formed_head
       tagged_immediates).
  - exact well_formed_input_to_head.
  - intros statics v Himmediate.
    split; [ exact (immediate_kind _ Himmediate) | ].
    split; [ | exact Himmediate ].
    destruct Himmediate as [i ->]. discriminate.
  - exact meet_equals_tagged_immediates_value_sound.
Qed.

Lemma meet_equals_single_tagged_immediate_sound_proof :
  meet_shortcut_sound Provers.meet_equals_single_tagged_immediate
    well_formed_input tagged_immediates
    (fun _ _ imm v => v = Value_const (Tagged_immediate imm)).
Proof.
  intros statics valuation tenv t v Hsat Hinput Hv Himmediate.
  pose proof
    (meet_equals_tagged_immediates_sound_proof _ _ _ _ _ Hsat Hinput Hv
       Himmediate)
    as Hsound.
  unfold Provers.meet_equals_single_tagged_immediate.
  destruct (Provers.meet_equals_tagged_immediates tenv t)
    as [[imms | | ] | ];
    try exact I; try exact Hsound.
  destruct (Keyed.get_singleton Z.eqb imms) as [imm | ] eqn:Hsingleton;
    [ | exact I ].
  destruct Hsound as [i [-> Hin]].
  rewrite (get_singleton_in _ _ _ Hsingleton Hin). reflexivity.
Qed.

Lemma meet_equals_single_tagged_immediate_silent :
  meet_shortcut_silent_on_poison Provers.meet_equals_single_tagged_immediate.
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hknown.
  unfold Provers.meet_equals_single_tagged_immediate in Hknown.
  destruct (Provers.meet_equals_tagged_immediates tenv t)
    as [[imms | | ] | ] eqn:Hmeet;
    try discriminate Hknown.
  exact
    (gen_value_to_meet_silent _ _ _ _ _ _ _ Hsat Hv Hpoison Hmeet).
Qed.

(* ======================================================================== *)
(* Naked numbers                                                            *)
(* ======================================================================== *)

Lemma expanded_naked_number_cases_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t : flambda_type)
    (k : naked_number_kind)
    (oub : or_unknown_or_bottom (list Z)) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_naked_number k oub) ->
  match oub with
  | Unknown => True
  | Bottom => False
  | Ok numbers => naked_numbers_satisfied k v numbers
  end.
Proof.
  intros Hsat Hok Hv Hd.
  pose proof
    (expand_head_descr_sound_in _ _ _ _ _ _ Hsat Hok
       (scrutinees_ok_naked_number _ _ _ _) Hv Hd)
    as Hexp.
  simpl in Hexp. rewrite type_satisfied_naked_number in Hexp.
  destruct oub as [ | numbers | ]; simpl in Hexp.
  - exact I.
  - exact Hexp.
  - exact Hexp.
Qed.

Lemma expanded_naked_number_cases (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (k : naked_number_kind)
    (oub : or_unknown_or_bottom (list Z)) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.expand_head tenv t = Some (Oub_naked_number k oub) ->
  match oub with
  | Unknown => True
  | Bottom => False
  | Ok numbers => naked_numbers_satisfied k v numbers
  end.
Proof.
  intros Hsat.
  exact
    (expanded_naked_number_cases_in (m := Strict) statics valuation tenv t k
       oub v Hsat I).
Qed.

Lemma meet_naked_number_sound_proof_in {m : mode} (kind : naked_number_kind) :
  kind <> Kind_naked_immediate ->
  meet_shortcut_sound_in m (Provers.meet_naked_number kind) any_input
    (values_of_kind (Naked_number kind))
    (fun _ _ numbers v => naked_numbers_satisfied kind v numbers).
Proof.
  intros Hkind_not statics valuation tenv t v Hsat Hok _ Hv Hkind.
  unfold Provers.meet_naked_number.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind_in _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  unfold values_of_kind in Hkind. rewrite Hkind in Hk.
  destruct d as [oub | oub | k oub | oub | oub]; simpl in Hk;
    try discriminate Hk.
  - injection Hk as ->. contradiction.
  - injection Hk as <-. rewrite kind_eqb_refl.
    pose proof (expanded_naked_number_cases_in _ _ _ _ _ _ _ Hsat Hok Hv Hd)
      as Hcases.
    destruct oub as [ | numbers | ]; [ exact I | | exact Hcases ].
    destruct (Keyed.is_empty numbers) eqn:Hempty; [ | exact Hcases ].
    rewrite (is_empty_nil _ Hempty) in Hcases.
    destruct Hcases as [c [n [_ [_ [_ []]]]]].
Qed.

Lemma meet_naked_number_sound_proof (kind : naked_number_kind) :
  kind <> Kind_naked_immediate ->
  meet_shortcut_sound (Provers.meet_naked_number kind) any_input
    (values_of_kind (Naked_number kind))
    (fun _ _ numbers v => naked_numbers_satisfied kind v numbers).
Proof.
  intros Hkind.
  exact
    (meet_shortcut_sound_strictly
       (meet_naked_number_sound_proof_in (m := Strict) kind Hkind)).
Qed.

Lemma meet_naked_number_silent (kind : naked_number_kind) :
  meet_shortcut_silent_on_poison (Provers.meet_naked_number kind).
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hknown.
  unfold Provers.meet_naked_number in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hknown.
  destruct (kind_eqb (Naked_number k) (Naked_number kind));
    [ | discriminate Hknown ].
  pose proof (expanded_naked_number_cases _ _ _ _ _ _ _ Hsat Hv Hd)
    as Hcases.
  destruct oub as [ | numbers | ]; try discriminate Hknown.
  destruct Hcases as [c [n [-> [_ [Hnumber _]]]]].
  destruct Hpoison as [k' Hpoison]. injection Hpoison as ->.
  discriminate Hnumber.
Qed.

(* ======================================================================== *)
(* A proof that is made of a prover on heads of which only [Proved] is      *)
(* sound                                                                    *)
(* ======================================================================== *)

(* The part of [generic_value_sound] that a proof needs: the answer
   [Proved] holds of every value of the head. *)
Definition generic_value_proved_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) (v : value) (x : A),
    typing_env_satisfied_strictly statics valuation tenv ->
    input tenv head ->
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    holds statics valuation x v.

Definition generic_value_proved_sound_in (m : mode) {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_value_non_null) (v : value) (x : A),
    sat m statics valuation tenv ->
    input tenv head ->
    head_of_kind_value_non_null_satisfied statics valuation v head ->
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    holds statics valuation x v.

Lemma gen_value_to_proof_proved_sound_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  generic_value_proved_sound_in m prove_gen input_head holds ->
  prover_sound_in m (Provers.gen_value_to_proof prove_gen) input holds.
Proof.
  intros Hinput Hgen statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.gen_value_to_proof in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
  destruct (prove_gen tenv head) as [[y | | ] | ] eqn:Hgen_eq;
    try discriminate Hproved.
  injection Hproved as ->.
  exact
    (Hgen statics valuation tenv head v x Hsat (Hinput _ _ _ _ Hin Hd) Hhead
       Hgen_eq).
Qed.

Lemma gen_value_to_proof_proved_sound {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (input_head : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  input_to_head input input_head ->
  generic_value_proved_sound prove_gen input_head holds ->
  prover_sound (Provers.gen_value_to_proof prove_gen) input holds.
Proof.
  intros Hinput Hgen.
  exact
    (prover_sound_strictly
       (gen_value_to_proof_proved_sound_in (m := Strict) prove_gen input
          input_head holds Hinput Hgen)).
Qed.

(* ======================================================================== *)
(* Aliases                                                                  *)
(* ======================================================================== *)

Lemma canonical_simple_of_alias_sound_in {m : mode} (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (t : flambda_type) (canonical : simple) (v : value) :
  sat m statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.canonical_simple_of_alias tenv min_name_mode t
  = Some (Some canonical) ->
  eval_simple valuation canonical = Some v.
Proof.
  intros Hsat Hv. unfold Provers.canonical_simple_of_alias.
  destruct (sat_but m _ _ _ Hsat) as [exempt Hbut].
  destruct (get_alias_opt t) as [s | ] eqn:Halias; [ | discriminate ].
  intros Hcanonical.
  rewrite
    (get_canonical_simple_satisfied exempt _ _ _ _ _ _ _ Hbut Hcanonical).
  exact (proj1 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hv)).
Qed.

Lemma canonical_simple_of_alias_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (t : flambda_type) (canonical : simple) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.canonical_simple_of_alias tenv min_name_mode t
  = Some (Some canonical) ->
  eval_simple valuation canonical = Some v.
Proof.
  strictly
    (canonical_simple_of_alias_sound_in (m := Strict) statics valuation tenv
       min_name_mode t canonical v).
Qed.

Lemma prove_equals_to_simple_of_kind_sound_proof_in {m : mode}
    (k : flambda_kind) :
  prover_sound_in m
    (fun tenv t => Provers.prove_equals_to_simple_of_kind tenv t k)
    any_input
    (fun _ valuation s v => eval_simple valuation s = Some v).
Proof.
  intros statics valuation tenv t s v Hsat Hok _ Hv Hproved.
  unfold Provers.prove_equals_to_simple_of_kind in Hproved.
  destruct (negb (kind_eqb (kind t) k)); [ discriminate Hproved | ].
  destruct (Provers.canonical_simple_of_alias tenv Name_mode.normal t)
    as [[canonical | ] | ] eqn:Hcanonical;
    try discriminate Hproved.
  injection Hproved as <-.
  exact (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hv Hcanonical).
Qed.

Lemma prove_equals_to_simple_of_kind_sound_proof (k : flambda_kind) :
  prover_sound
    (fun tenv t => Provers.prove_equals_to_simple_of_kind tenv t k)
    any_input
    (fun _ valuation s v => eval_simple valuation s = Some v).
Proof.
  strictly (prove_equals_to_simple_of_kind_sound_proof_in (m := Strict) k).
Qed.

(* The number of the immediates of a variant, as a [Simple]. *)
Lemma inspect_tagging_immediates_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (immediates : Or_unknown.t flambda_type) (i : Z) :
  typing_env_satisfied_strictly statics valuation tenv ->
  match immediates with
  | Or_unknown.Unknown => True
  | Or_unknown.Known ty =>
    well_formed_input tenv ty
    ∧ type_satisfied statics valuation (Value_const (Naked_immediate i)) ty
  end ->
  match Provers.inspect_tagging_immediates min_name_mode tenv immediates with
  | Some (Generic_proof.Proved s) =>
    eval_simple valuation s = Some (Value_const (Naked_immediate i))
  | Some Generic_proof.Invalid => False
  | Some Generic_proof.Unknown | None => True
  end.
Proof.
  intros Hsat Himms. unfold Provers.inspect_tagging_immediates.
  destruct immediates as [ty | ]; [ | exact I ].
  destruct Himms as [Hinput Hty].
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[canonical | ] | ] eqn:Hcanonical;
    [ | | exact I ].
  - exact (canonical_simple_of_alias_sound _ _ _ _ _ _ _ Hsat Hty Hcanonical).
  - pose proof
      (meet_naked_immediates_sound_proof _ _ _ _ _ Hsat Hinput Hty eq_refl)
      as Hmeet.
    destruct (Provers.meet_naked_immediates tenv ty) as [[imms | | ] | ];
      try exact I; try exact Hmeet.
    destruct (Keyed.get_singleton Z.eqb imms) as [imm | ] eqn:Hsingleton;
      [ | exact I ].
    destruct Hmeet as [i' [Heq Hin]]. injection Heq as <-.
    rewrite (get_singleton_in _ _ _ Hsingleton Hin). reflexivity.
Qed.

Lemma inspect_tagging_of_simple_value_sound
    (proof_kind : Provers.tagging_proof_kind) (min_name_mode : Name_mode.t) :
  generic_value_sound
    (Provers.inspect_tagging_of_simple_value proof_kind min_name_mode)
    well_formed_head
    (fun _ v =>
       match proof_kind with
       | Provers.Prove => True
       | Provers.Meet => is_immediate v
       end)
    (fun _ => tagging_of).
Proof.
  intros statics valuation tenv head v Hsat Hinput Hhead Hdomain.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try (destruct proof_kind; exact I).
  assert (Hinspect :
    (∃ i : Z,
      v = Value_const (Tagged_immediate i)
      ∧ match immediates with
        | Or_unknown.Unknown => True
        | Or_unknown.Known ty =>
          type_satisfied statics valuation (Value_const (Naked_immediate i))
            ty
        end) ->
    match
      Provers.inspect_tagging_immediates min_name_mode tenv immediates
    with
    | Some (Generic_proof.Proved s) => tagging_of valuation s v
    | Some Generic_proof.Invalid => False
    | Some Generic_proof.Unknown | None => True
    end).
  { intros [i [-> Himms]].
    pose proof
      (inspect_tagging_immediates_sound statics valuation tenv min_name_mode
         immediates i Hsat)
      as Hsound.
    match type of Hsound with
    | ?P -> _ => assert (Hpremise : P)
    end.
    { destruct immediates as [ty | ]; [ | exact I ].
      split; [ | exact Himms ].
      exact (well_formed_head_immediates _ _ _ _ _ _ _ Hinput). }
    specialize (Hsound Hpremise).
    destruct (Provers.inspect_tagging_immediates min_name_mode tenv immediates)
      as [[s | | ] | ];
      try exact I; try exact Hsound.
    exists i. split; [ reflexivity | exact Hsound ]. }
  destruct proof_kind.
  - destruct blocks as [blocks | ]; [ | exact I ].
    destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks; [ | exact I ].
    apply Hinspect.
    exact (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks).
  - assert (Hcase :
      match blocks with
      | Or_unknown.Unknown | Or_unknown.Known _ =>
        match
          Provers.inspect_tagging_immediates min_name_mode tenv immediates
        with
        | Some (Generic_proof.Proved s) => tagging_of valuation s v
        | Some Generic_proof.Invalid => False
        | Some Generic_proof.Unknown | None => True
        end
      end).
    { assert (Himmediate :
        ∃ i : Z,
          v = Value_const (Tagged_immediate i)
          ∧ match immediates with
            | Or_unknown.Unknown => True
            | Or_unknown.Known ty =>
              type_satisfied statics valuation
                (Value_const (Naked_immediate i)) ty
            end).
      { destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
          as [Himmediate | [tag [fields [Hblock _]]]];
          [ exact Himmediate | ].
        rewrite (immediate_not_block statics v Hdomain) in Hblock.
        discriminate Hblock. }
      destruct blocks; exact (Hinspect Himmediate). }
    destruct blocks; exact Hcase.
Qed.

Lemma prove_tagging_of_simple_sound_proof (min_name_mode : Name_mode.t) :
  prover_sound
    (fun tenv t => Provers.prove_tagging_of_simple tenv min_name_mode t)
    well_formed_input (fun _ => tagging_of).
Proof.
  apply (gen_value_to_proof_sound _ well_formed_input well_formed_head).
  - exact well_formed_input_to_head.
  - exact (inspect_tagging_of_simple_value_sound Provers.Prove min_name_mode).
Qed.

Lemma meet_tagging_of_simple_sound_proof (min_name_mode : Name_mode.t) :
  meet_shortcut_sound
    (fun tenv t => Provers.meet_tagging_of_simple tenv min_name_mode t)
    well_formed_input tagged_immediates (fun _ => tagging_of).
Proof.
  apply
    (gen_value_to_meet_sound _ well_formed_input well_formed_head
       (fun _ v => is_immediate v)).
  - exact well_formed_input_to_head.
  - intros statics v Himmediate.
    split; [ exact (immediate_kind _ Himmediate) | ].
    split; [ | exact Himmediate ].
    destruct Himmediate as [i ->]. discriminate.
  - exact (inspect_tagging_of_simple_value_sound Provers.Meet min_name_mode).
Qed.

(* ======================================================================== *)
(* Boxed numbers                                                            *)
(* ======================================================================== *)

(* A boxed number of a kind has the head of the boxed numbers of that
   kind of which nothing more is known. *)
Lemma boxed_head_satisfied (statics : static_env) (vars : env) (v : value)
    (k : boxable_number) :
  boxed_numbers k statics v ->
  head_of_kind_value_non_null_satisfied statics vars v
    (Boxed_number k (unknown (unboxed_kind k)) Alloc_mode.Heap).
Proof.
  intros [c [Hboxed Hkind]]. rewrite boxed_number_satisfied.
  exists c. split; [ exact Hboxed | ]. split; [ exact Hkind | ].
  apply unknown_satisfied. exact Hkind.
Qed.

(* A head of another form, or of the boxed numbers of another kind,
   has no boxed number of the kind. *)
Lemma not_boxed_head (statics : static_env) (vars : env) (v : value)
    (k : boxable_number) (head : head_of_kind_value_non_null) :
  head_of_kind_value_non_null_satisfied statics vars v head ->
  boxed_numbers k statics v ->
  match head with
  | Boxed_number k' _ _ => k = k'
  | Variant _ _ _ _ _ _ | Mutable_block _ | Closures _ _ | String _ => False
  end.
Proof.
  intros Hhead Hboxed.
  pose proof
    (heads_compatible_satisfied _ _ _ _ _
       (boxed_head_satisfied statics vars v k Hboxed) Hhead)
    as Hcompatible.
  destruct head; exact Hcompatible.
Qed.

Lemma prove_is_a_boxed_or_tagged_number_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_is_a_boxed_or_tagged_number_value
    any_head (fun _ _ => True) boxed_or_tagged_number_is.
Proof.
  intros statics valuation tenv head v Hsat _ Hhead _.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I.
  - destruct blocks as [blocks | ]; [ | exact I ].
    destruct (row_like_for_blocks_is_bottom blocks) eqn:Hblocks; [ | exact I ].
    destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hblocks)
      as [i [-> _]].
    exists i. reflexivity.
  - rewrite boxed_number_satisfied in Hhead. exact Hhead.
Qed.

Lemma prove_is_a_boxed_or_tagged_number_value_sound :
  generic_value_sound Provers.prove_is_a_boxed_or_tagged_number_value
    any_head (fun _ _ => True) boxed_or_tagged_number_is.
Proof.
  strictly (prove_is_a_boxed_or_tagged_number_value_sound_in (m := Strict)).
Qed.

Lemma prove_is_a_boxed_or_tagged_number_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_is_a_boxed_or_tagged_number any_input
    boxed_or_tagged_number_is.
Proof.
  apply (gen_value_to_proof_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_is_a_boxed_or_tagged_number_value_sound_in.
Qed.

Lemma prove_is_a_boxed_or_tagged_number_sound_proof :
  prover_sound Provers.prove_is_a_boxed_or_tagged_number any_input
    boxed_or_tagged_number_is.
Proof.
  strictly (prove_is_a_boxed_or_tagged_number_sound_proof_in (m := Strict)).
Qed.

Lemma prove_is_a_tagged_immediate_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_is_a_tagged_immediate any_input
    (fun _ _ _ v => is_immediate v).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.prove_is_a_tagged_immediate in Hproved.
  destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
    as [[[alloc_mode k contents | ] | ] | ] eqn:Hnumber;
    try discriminate Hproved.
  exact
    (prove_is_a_boxed_or_tagged_number_sound_proof_in _ _ _ _ _ _ Hsat Hok Hin
       Hv Hnumber).
Qed.

Lemma prove_is_a_tagged_immediate_sound_proof :
  prover_sound Provers.prove_is_a_tagged_immediate any_input
    (fun _ _ _ v => is_immediate v).
Proof.
  strictly (prove_is_a_tagged_immediate_sound_proof_in (m := Strict)).
Qed.

Lemma prove_is_a_boxed_number_sound_proof_in {m : mode}
    (kind : boxable_number) :
  prover_sound_in m (Provers.prove_is_a_boxed_number kind) any_input
    (fun statics _ _ v => boxed_numbers kind statics v).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.prove_is_a_boxed_number in Hproved.
  destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
    as [[[alloc_mode k contents | ] | ] | ] eqn:Hnumber;
    try discriminate Hproved.
  destruct (boxable_number_eqb k kind) eqn:Hk; [ | discriminate Hproved ].
  apply boxable_number_eqb_eq in Hk. subst k.
  destruct
    (prove_is_a_boxed_or_tagged_number_sound_proof_in _ _ _ _ _ _ Hsat Hok Hin
       Hv Hnumber)
    as [c [Hboxed [Hkind _]]].
  exists c. split; assumption.
Qed.

Lemma prove_is_a_boxed_number_sound_proof (kind : boxable_number) :
  prover_sound (Provers.prove_is_a_boxed_number kind) any_input
    (fun statics _ _ v => boxed_numbers kind statics v).
Proof.
  strictly (prove_is_a_boxed_number_sound_proof_in (m := Strict) kind).
Qed.

Lemma prove_is_or_is_not_a_boxed_float_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_is_or_is_not_a_boxed_float_value
    any_head (fun _ _ => True)
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead _.
  pose proof (not_boxed_head _ _ _ Boxable_naked_float _ Hhead) as Hnot.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact Hnot.
  destruct k; simpl;
    try (intros Hboxed; discriminate (Hnot Hboxed)).
  rewrite boxed_number_satisfied in Hhead.
  destruct Hhead as [c [Hboxed [Hkind _]]].
  exists c. split; assumption.
Qed.

Lemma prove_is_or_is_not_a_boxed_float_value_sound :
  generic_value_sound Provers.prove_is_or_is_not_a_boxed_float_value
    any_head (fun _ _ => True)
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).
Proof.
  strictly (prove_is_or_is_not_a_boxed_float_value_sound_in (m := Strict)).
Qed.

Lemma prove_is_or_is_not_a_boxed_float_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_is_or_is_not_a_boxed_float any_input
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).
Proof.
  apply (gen_value_to_proof_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_is_or_is_not_a_boxed_float_value_sound_in.
Qed.

Lemma prove_is_or_is_not_a_boxed_float_sound_proof :
  prover_sound Provers.prove_is_or_is_not_a_boxed_float any_input
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).
Proof.
  strictly (prove_is_or_is_not_a_boxed_float_sound_proof_in (m := Strict)).
Qed.

Lemma boxed_number_kind_not_null (statics : static_env) (v : value)
    (k : boxable_number) :
  boxed_numbers k statics v ->
  kind_of_value v = Value ∧ v <> Value_const Null.
Proof.
  intros [c [Hboxed _]].
  split; [ exact (boxed_number_of_kind _ _ _ Hboxed) | ].
  intros ->. discriminate Hboxed.
Qed.

Lemma meet_boxed_number_containing_simple_sound_proof_in {m : mode}
    (kind : boxable_number) (min_name_mode : Name_mode.t) :
  meet_shortcut_sound_in m
    (fun tenv t =>
       Provers.meet_boxed_number_containing_simple kind tenv min_name_mode t)
    any_input (boxed_numbers kind)
    (fun statics valuation s v =>
       ∃ c : const,
         boxed_number_of statics v = Some c
         ∧ eval_simple valuation s = Some (Value_const c)).
Proof.
  intros statics valuation tenv t v Hsat Hok _ Hv Hboxed.
  destruct (boxed_number_kind_not_null _ _ _ Hboxed) as [Hkind Hnot_null].
  unfold Provers.meet_boxed_number_containing_simple.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd; [ | exact I ].
  pose proof (expanded_kind_in _ _ _ _ _ _ Hsat Hv Hd) as Hk.
  rewrite Hkind in Hk.
  destruct d as [oub | oub | k oub | oub | oub]; try discriminate Hk.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  destruct oub as [ | [[ | head | ] is_null] | ]; simpl in Hcases |- *.
  - exact I.
  - exact I.
  - destruct Hcases as [[Hnull _] | [_ Hhead]]; [ contradiction | ].
    pose proof (not_boxed_head _ _ _ _ _ Hhead Hboxed) as Hnot.
    destruct head as [is_int immediates get_tag blocks extensions is_unique
                     | alloc_mode | k contents alloc_mode
                     | by_function_slot alloc_mode | strs];
      try exact Hnot.
    subst k.
    rewrite (proj2 (boxable_number_eqb_eq kind kind) eq_refl).
    rewrite boxed_number_satisfied in Hhead.
    destruct Hhead as [c [Hc [_ Hcontents]]].
    destruct (Provers.canonical_simple_of_alias tenv min_name_mode contents)
      as [[canonical | ] | ] eqn:Hcanonical;
      try exact I.
    exists c. split; [ exact Hc | ].
    exact
      (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hcontents
         Hcanonical).
  - destruct Hcases as [[Hnull _] | [_ []]]. contradiction.
  - exact Hcases.
Qed.

Lemma meet_boxed_number_containing_simple_sound_proof
    (kind : boxable_number) (min_name_mode : Name_mode.t) :
  meet_shortcut_sound
    (fun tenv t =>
       Provers.meet_boxed_number_containing_simple kind tenv min_name_mode t)
    any_input (boxed_numbers kind)
    (fun statics valuation s v =>
       ∃ c : const,
         boxed_number_of statics v = Some c
         ∧ eval_simple valuation s = Some (Value_const c)).
Proof.
  strictly
    (meet_boxed_number_containing_simple_sound_proof_in (m := Strict) kind
       min_name_mode).
Qed.

Lemma meet_boxed_number_containing_simple_silent (kind : boxable_number)
    (min_name_mode : Name_mode.t) :
  meet_shortcut_silent_on_poison
    (fun tenv t =>
       Provers.meet_boxed_number_containing_simple kind tenv min_name_mode t).
Proof.
  intros statics valuation tenv t x v Hsat Hv Hpoison Hknown.
  unfold Provers.meet_boxed_number_containing_simple in Hknown.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] is_null] | ] | oub | k oub | oub | oub];
    try discriminate Hknown.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[Hnull _] | [_ Hhead]].
  - exact (poison_not_null _ Hpoison Hnull).
  - exact (poison_no_head _ _ _ _ Hpoison Hhead).
Qed.

(* ======================================================================== *)
(* Strings                                                                  *)
(* ======================================================================== *)

Lemma string_head_satisfied (statics : static_env) (vars : env) (v : value)
    (bytes : list Z) :
  string_of statics v = Some bytes ->
  head_of_kind_value_non_null_satisfied statics vars v (String [bytes]).
Proof.
  intros Hstring. rewrite string_satisfied.
  exists bytes. split; [ exact Hstring | left; reflexivity ].
Qed.

Lemma prove_strings_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_strings_value any_head strings
    (fun statics _ => string_among statics).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead [bytes Hstring].
  pose proof
    (heads_compatible_satisfied _ _ _ _ _
       (string_head_satisfied statics valuation v bytes Hstring) Hhead)
    as Hcompatible.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact Hcompatible.
  rewrite string_satisfied in Hhead. exact Hhead.
Qed.

Lemma prove_strings_value_sound :
  generic_value_sound Provers.prove_strings_value any_head strings
    (fun statics _ => string_among statics).
Proof.
  strictly (prove_strings_value_sound_in (m := Strict)).
Qed.

Lemma prove_strings_value_proved_sound_in {m : mode} :
  generic_value_proved_sound_in m Provers.prove_strings_value any_head
    (fun statics _ => string_among statics).
Proof.
  intros statics valuation tenv head v x Hsat _ Hhead Hproved.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try discriminate Hproved.
  injection Hproved as <-.
  rewrite string_satisfied in Hhead. exact Hhead.
Qed.

Lemma prove_strings_value_proved_sound :
  generic_value_proved_sound Provers.prove_strings_value any_head
    (fun statics _ => string_among statics).
Proof.
  strictly (prove_strings_value_proved_sound_in (m := Strict)).
Qed.

Lemma prove_strings_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_strings any_input
    (fun statics _ => string_among statics).
Proof.
  apply (gen_value_to_proof_proved_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_strings_value_proved_sound_in.
Qed.

Lemma prove_strings_sound_proof :
  prover_sound Provers.prove_strings any_input
    (fun statics _ => string_among statics).
Proof.
  strictly (prove_strings_sound_proof_in (m := Strict)).
Qed.

Lemma meet_strings_sound_proof_in {m : mode} :
  meet_shortcut_sound_in m Provers.meet_strings any_input strings
    (fun statics _ => string_among statics).
Proof.
  apply (gen_value_to_meet_sound_in _ any_input any_head strings).
  - exact any_input_to_head.
  - intros statics v [bytes Hstring].
    split; [ exact (string_of_kind _ _ _ Hstring) | ].
    split; [ | exists bytes; exact Hstring ].
    intros ->. discriminate Hstring.
  - exact prove_strings_value_sound_in.
Qed.

Lemma meet_strings_sound_proof :
  meet_shortcut_sound Provers.meet_strings any_input strings
    (fun statics _ => string_among statics).
Proof.
  strictly (meet_strings_sound_proof_in (m := Strict)).
Qed.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Lemma variant_head_variants (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Variant is_int immediates get_tag blocks extensions is_unique) ->
  variants statics v.
Proof.
  intros Hhead.
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
    as [[i [-> _]] | [tag [fields [Hblock _]]]].
  - left. exists i. reflexivity.
  - right. exists tag, fields. exact Hblock.
Qed.

(* A prover on heads that answers [Proved] of variants only, and is
   sound of the values that are a tagged immediate or a block, is
   sound as a proof. *)
Lemma generic_value_sound_proved_of_variants_in {m : mode} {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  generic_value_sound_in m prove_gen input variants holds ->
  (∀ (tenv : Typing_env.t) (head : head_of_kind_value_non_null) (x : A),
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    match head with
    | Variant _ _ _ _ _ _ => True
    | _ => False
    end) ->
  generic_value_proved_sound_in m prove_gen input holds.
Proof.
  intros Hgen Hvariant statics valuation tenv head v x Hsat Hin Hhead Hproved.
  pose proof (Hvariant _ _ _ Hproved) as His_variant.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try contradiction.
  pose proof
    (Hgen statics valuation tenv _ v Hsat Hin Hhead
       (variant_head_variants _ _ _ _ _ _ _ _ _ Hhead))
    as Hsound.
  rewrite Hproved in Hsound. exact Hsound.
Qed.

Lemma generic_value_sound_proved_of_variants {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (input : Typing_env.t -> head_of_kind_value_non_null -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) :
  generic_value_sound prove_gen input variants holds ->
  (∀ (tenv : Typing_env.t) (head : head_of_kind_value_non_null) (x : A),
    prove_gen tenv head = Some (Generic_proof.Proved x) ->
    match head with
    | Variant _ _ _ _ _ _ => True
    | _ => False
    end) ->
  generic_value_proved_sound prove_gen input holds.
Proof.
  strictly
    (generic_value_sound_proved_of_variants_in (m := Strict) prove_gen input
       holds).
Qed.

Lemma prove_variant_like_generic_value_sound :
  generic_value_sound Provers.prove_variant_like_generic_value
    well_formed_head variants (fun statics _ => variant_like statics).
Proof.
  intros statics valuation tenv head v Hsat Hinput Hhead Hdomain.
  pose proof (not_variant_head _ _ _ _ Hhead) as Hnot_variant.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I; try exact (Hnot_variant Hdomain).
  destruct blocks as [blocks | ]; [ | exact I ].
  destruct (row_like_for_blocks_all_tags_and_sizes blocks) as [sizes | ]
    eqn:Hsizes;
    [ | exact I ].
  destruct
    (Keyed.for_all_bindings_b Z.eqb
       (fun (tag : Z) (_ : nat * block_shape) =>
          Provers.tag_is_scannable tag)
       sizes);
    [ | exact I ].
  (* What is to be shown of the constant constructors answered. *)
  assert (Hproof :
    ∀ const_ctors : Or_unknown.t (list Z),
      (∀ i : Z,
        v = Value_const (Tagged_immediate i) ->
        match immediates with
        | Or_unknown.Unknown => True
        | Or_unknown.Known ty =>
          type_satisfied statics valuation (Value_const (Naked_immediate i))
            ty
        end ->
        match const_ctors with
        | Or_unknown.Known imms => i ∈ imms
        | Or_unknown.Unknown => True
        end) ->
      variant_like statics (Variant_like_proof const_ctors sizes) v).
  { intros const_ctors Hconst.
    destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
      as [[i [Hv Himms]] | [tag [fields [Hblock Hblocks]]]].
    - left. exists i. split; [ exact Hv | exact (Hconst i Hv Himms) ].
    - right.
      destruct
        (all_tags_and_sizes_satisfied _ _ _ _ _ _ Hsizes Hblocks)
        as [size [shape [Hlookup [Hlength Hshape]]]].
      exists tag, fields, size, shape. auto. }
  destruct immediates as [imms | ].
  - pose proof
      (fun i Hty =>
         prove_naked_immediates_generic_sound statics valuation tenv imms
           (Value_const (Naked_immediate i)) Hsat
           (well_formed_head_immediates _ _ _ _ _ _ _ Hinput) Hty eq_refl)
      as Hsound.
    destruct (Provers.prove_naked_immediates_generic tenv imms)
      as [[imms' | | ] | ];
      simpl; try exact I; apply Hproof; intros i Hv Hty.
    + destruct (Hsound i Hty) as [i' [Heq Hin]]. injection Heq as <-.
      exact Hin.
    + exact I.
    + destruct (Hsound i Hty).
  - simpl. apply Hproof. intros i _ _. exact I.
Qed.

Lemma prove_variant_like_sound_proof :
  prover_sound Provers.prove_variant_like well_formed_input
    (fun statics _ => variant_like statics).
Proof.
  apply (gen_value_to_proof_proved_sound _ well_formed_input well_formed_head).
  - exact well_formed_input_to_head.
  - apply generic_value_sound_proved_of_variants.
    + exact prove_variant_like_generic_value_sound.
    + intros tenv head x.
      destruct head; simpl; first [ intros _; exact I | discriminate ].
Qed.

Lemma meet_variant_like_sound_proof :
  meet_shortcut_sound Provers.meet_variant_like well_formed_input variants
    (fun statics _ => variant_like statics).
Proof.
  apply
    (gen_value_to_meet_sound _ well_formed_input well_formed_head variants).
  - exact well_formed_input_to_head.
  - intros statics v Hvariant.
    split; [ | split; [ | exact Hvariant ] ].
    + destruct Hvariant as [Himmediate | Hblock].
      * exact (immediate_kind _ Himmediate).
      * exact (block_kind _ _ Hblock).
    + destruct Hvariant as [[i ->] | [tag [fields Hblock]]].
      * discriminate.
      * intros ->. discriminate Hblock.
  - exact prove_variant_like_generic_value_sound.
Qed.

(* What [Provers.prove_unique_tag_and_size_value] answers: the block,
   and the types of its fields. *)
Definition unique_block (statics : static_env) (valuation : env)
    (x : Z * block_shape * nat * list flambda_type * Alloc_mode.t)
    (v : value) : Prop :=
  let '(tag, shape, size, tys, _) := x in
  ∃ fields : list value,
    block_of statics v = Some (tag, fields)
    ∧ length fields = size
    ∧ block_has_shape shape tag fields
    ∧ fields_satisfied (type_satisfied statics valuation) tys fields.

Lemma prove_unique_tag_and_size_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_unique_tag_and_size_value any_head
    (fun _ _ => True) unique_block.
Proof.
  intros statics valuation tenv head v Hsat _ Hhead _.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I.
  destruct immediates as [imms | ]; [ | exact I ].
  destruct (Provers.is_bottom tenv imms) as [[ | ] | ] eqn:Himms;
    simpl; try exact I.
  destruct (variant_no_immediates_in _ _ _ _ _ _ _ _ _ _ Hsat Hhead Himms)
    as [_ [tag' [fields [Hblock Hblocks]]]].
  destruct blocks as [blocks | ]; [ | exact I ].
  destruct (row_like_for_blocks_get_singleton blocks)
    as [[[[[tag shape] size] tys] alloc_mode] | ] eqn:Hsingleton;
    [ | exact I ].
  destruct
    (get_singleton_satisfied _ _ _ _ _ _ _ _ _ _ Hsingleton Hblocks)
    as [-> [Hlength [Hshape Hfields]]].
  exists fields. auto.
Qed.

Lemma prove_unique_tag_and_size_value_sound :
  generic_value_sound Provers.prove_unique_tag_and_size_value any_head
    (fun _ _ => True) unique_block.
Proof.
  strictly (prove_unique_tag_and_size_value_sound_in (m := Strict)).
Qed.

Lemma prove_unique_tag_and_size_value_proof_sound_in {m : mode} :
  prover_sound_in m
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value)
    any_input unique_block.
Proof.
  apply (gen_value_to_proof_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_unique_tag_and_size_value_sound_in.
Qed.

Lemma prove_unique_tag_and_size_value_proof_sound :
  prover_sound
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value)
    any_input unique_block.
Proof.
  strictly (prove_unique_tag_and_size_value_proof_sound_in (m := Strict)).
Qed.

Lemma prove_unique_tag_and_size_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_unique_tag_and_size any_input
    (fun statics _ => block_of_tag_shape_and_size statics).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.prove_unique_tag_and_size in Hproved.
  destruct
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value tenv
       t)
    as [[[[[[tag shape] size] tys] alloc_mode] | ] | ] eqn:Hgen;
    try discriminate Hproved.
  injection Hproved as <-.
  destruct
    (prove_unique_tag_and_size_value_proof_sound_in _ _ _ _ _ _ Hsat Hok Hin Hv
       Hgen)
    as [fields [Hblock [Hlength [Hshape _]]]].
  exists fields. auto.
Qed.

Lemma prove_unique_tag_and_size_sound_proof :
  prover_sound Provers.prove_unique_tag_and_size any_input
    (fun statics _ => block_of_tag_shape_and_size statics).
Proof.
  strictly (prove_unique_tag_and_size_sound_proof_in (m := Strict)).
Qed.

Lemma field_aliases_fold_unknown (tys : list flambda_type) :
  List.fold_left Provers.field_aliases_rev tys Proof_of_property.Unknown
  = Proof_of_property.Unknown.
Proof. induction tys as [ | ty tys IH]; [ reflexivity | exact IH ]. Qed.

Lemma field_aliases_fold (tys : list flambda_type) :
  ∀ acc simples_rev : list simple,
    List.fold_left Provers.field_aliases_rev tys
      (Proof_of_property.Proved acc)
    = Proof_of_property.Proved simples_rev ->
    ∃ aliases : list simple,
      simples_rev = rev aliases ++ acc
      ∧ Forall2 (fun ty s => get_alias_opt ty = Some s) tys aliases.
Proof.
  induction tys as [ | ty tys IH]; intros acc simples_rev Hfold;
    simpl in Hfold.
  - injection Hfold as <-. exists []. split; [ reflexivity | constructor ].
  - destruct (get_alias_opt ty) as [s | ] eqn:Halias.
    + destruct (IH _ _ Hfold) as [aliases [-> Hall]].
      exists (s :: aliases). split.
      * simpl. rewrite <- app_assoc. reflexivity.
      * constructor; assumption.
    + rewrite field_aliases_fold_unknown in Hfold. discriminate Hfold.
Qed.

Lemma forall2_nth_right {A B : Type} (P : A -> B -> Prop) (l : list A)
    (l' : list B) :
  Forall2 P l l' ->
  ∀ (i : nat) (b : B),
    nth_error l' i = Some b ->
    ∃ a : A, nth_error l i = Some a ∧ P a b.
Proof.
  induction 1 as [ | a b' l l' Hab Hall IH]; intros i b Hnth.
  - destruct i; discriminate Hnth.
  - destruct i as [ | i]; simpl in Hnth.
    + injection Hnth as <-. exists a. split; [ reflexivity | exact Hab ].
    + exact (IH i b Hnth).
Qed.

Lemma prove_unique_immutable_heap_block_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_unique_fully_constructed_immutable_heap_block
    any_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ ∀ (i : nat) (s : simple) (field : value),
             nth_error simples i = Some s ->
             nth_error fields i = Some field ->
             eval_simple valuation s = Some field).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.prove_unique_fully_constructed_immutable_heap_block
    in Hproved.
  destruct
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value tenv
       t)
    as [[[[[[tag shape] size] tys] alloc_mode] | ] | ] eqn:Hgen;
    try discriminate Hproved.
  destruct alloc_mode; try discriminate Hproved.
  destruct
    (List.fold_left Provers.field_aliases_rev tys
       (Proof_of_property.Proved []))
    as [simples_rev | ] eqn:Hfold;
    try discriminate Hproved.
  injection Hproved as <-.
  destruct (field_aliases_fold _ _ _ Hfold) as [aliases [-> Haliases]].
  rewrite app_nil_r, rev_involutive.
  destruct
    (prove_unique_tag_and_size_value_proof_sound_in _ _ _ _ _ _ Hsat Hok Hin Hv
       Hgen)
    as [fields [Hblock [Hlength [Hshape Hfields]]]].
  exists fields. split; [ exact Hblock | ]. split; [ exact Hlength | ].
  split; [ exact Hshape | ].
  intros i s field Hs Hfield.
  destruct (forall2_nth_right _ _ _ Haliases i s Hs) as [ty [Hty Halias]].
  pose proof
    (proj1 (fields_satisfied_nth _ _ _) Hfields i ty field Hty Hfield)
    as Hsatisfied.
  exact (proj1 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hsatisfied)).
Qed.

Lemma prove_unique_immutable_heap_block_sound_proof :
  prover_sound Provers.prove_unique_fully_constructed_immutable_heap_block
    any_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ ∀ (i : nat) (s : simple) (field : value),
             nth_error simples i = Some s ->
             nth_error fields i = Some field ->
             eval_simple valuation s = Some field).
Proof.
  strictly (prove_unique_immutable_heap_block_sound_proof_in (m := Strict)).
Qed.

(* With a well-formed head, the types of the fields that
   [Provers.prove_unique_tag_and_size_value] answers are as many as
   the size. *)
Lemma prove_unique_tag_and_size_value_length_in {m : mode} :
  generic_value_proved_sound_in m Provers.prove_unique_tag_and_size_value
    well_formed_head
    (fun _ _ x _ => let '(_, _, size, tys, _) := x in length tys = size).
Proof.
  intros statics valuation tenv head v x Hsat [_ Hwf] Hhead Hproved.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try discriminate Hproved.
  simpl in Hproved.
  destruct immediates as [imms | ]; [ | discriminate Hproved ].
  destruct (Provers.is_bottom tenv imms) as [[ | ] | ];
    simpl in Hproved; try discriminate Hproved.
  destruct blocks as [blocks | ]; [ | discriminate Hproved ].
  destruct (row_like_for_blocks_get_singleton blocks)
    as [[[[[tag shape] size] tys] alloc_mode] | ] eqn:Hsingleton;
    [ | discriminate Hproved ].
  injection Hproved as <-.
  rewrite variant_well_formed in Hwf.
  apply andb_prop in Hwf. destruct Hwf as [Hwf _].
  apply andb_prop in Hwf. destruct Hwf as [_ Hwf].
  destruct blocks as [known_tags other_tags alloc_mode'].
  destruct (get_singleton_case _ _ _ _ _ _ _ _ Hsingleton)
    as [_ [case [Hlookup [Hdomain [_ Hmaps]]]]].
  specialize (Hlookup tag). rewrite Z.eqb_refl in Hlookup.
  pose proof
    (proj1
       (proj2
          (proj1
             (row_like_for_blocks_well_formed_iff known_tags other_tags
                alloc_mode')
             Hwf))
       tag case Hlookup)
    as Hcase.
  apply row_like_block_case_well_formed_iff in Hcase.
  destruct Hcase as [Hfits _].
  unfold block_product_fits in Hfits. rewrite Hdomain, Hmaps in Hfits.
  exact (proj1 (Nat.eqb_eq _ _) Hfits).
Qed.

Lemma prove_unique_tag_and_size_value_length :
  generic_value_proved_sound Provers.prove_unique_tag_and_size_value
    well_formed_head
    (fun _ _ x _ => let '(_, _, size, tys, _) := x in length tys = size).
Proof.
  strictly (prove_unique_tag_and_size_value_length_in (m := Strict)).
Qed.

Lemma eval_simples_of_fields (valuation : env) (simples : list simple)
    (fields : list value) :
  Forall2 (fun s field => eval_simple valuation s = Some field) simples
    fields ->
  eval_simples valuation simples = Some fields.
Proof.
  induction 1 as [ | s field simples fields Hs Hall IH];
    [ reflexivity | ].
  simpl. rewrite Hs. simpl. rewrite IH. reflexivity.
Qed.

Lemma forall2_compose {A B C : Type} (P : A -> B -> Prop)
    (Q : B -> C -> Prop) (R : C -> A -> Prop) (l : list A) (l' : list B) :
  (∀ (a : A) (b : B) (c : C), P a b -> Q b c -> R c a) ->
  Forall2 P l l' ->
  ∀ l'' : list C, Forall2 Q l' l'' -> Forall2 R l'' l.
Proof.
  intros Hcompose. induction 1 as [ | a b l l' Hab Hall IH];
    intros l'' Hq; inversion Hq; subst; constructor; eauto.
Qed.

Lemma prove_unique_immutable_heap_block_fields_proof_in {m : mode} :
  prover_sound_in m Provers.prove_unique_fully_constructed_immutable_heap_block
    well_formed_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ length simples = size
         ∧ eval_simples valuation simples = Some fields).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  unfold Provers.prove_unique_fully_constructed_immutable_heap_block
    in Hproved.
  destruct
    (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value tenv
       t)
    as [[[[[[tag shape] size] tys] alloc_mode] | ] | ] eqn:Hgen;
    try discriminate Hproved.
  destruct alloc_mode; try discriminate Hproved.
  destruct
    (List.fold_left Provers.field_aliases_rev tys
       (Proof_of_property.Proved []))
    as [simples_rev | ] eqn:Hfold;
    try discriminate Hproved.
  injection Hproved as <-.
  destruct (field_aliases_fold _ _ _ Hfold) as [aliases [-> Haliases]].
  rewrite app_nil_r, rev_involutive.
  destruct
    (prove_unique_tag_and_size_value_proof_sound_in _ _ _ _ _ _ Hsat Hok I Hv
       Hgen)
    as [fields [Hblock [Hlength [Hshape Hfields]]]].
  pose proof
    (gen_value_to_proof_proved_sound_in _ well_formed_input well_formed_head _
       well_formed_input_to_head prove_unique_tag_and_size_value_length_in _ _
       _ _ _ _ Hsat Hok Hin Hv Hgen)
    as Hsize.
  simpl in Hsize.
  exists fields. split; [ exact Hblock | ]. split; [ exact Hlength | ].
  split; [ exact Hshape | ].
  split; [ rewrite <- (Forall2_length Haliases); exact Hsize | ].
  apply eval_simples_of_fields.
  assert (Hlength' : length fields = length tys).
  { rewrite Hlength, Hsize. reflexivity. }
  apply (fields_satisfied_Forall2 _ _ _ Hlength') in Hfields.
  refine (forall2_compose _ _ _ _ _ _ Hfields aliases Haliases).
  intros field ty s Hsatisfied Halias.
  exact (proj1 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hsatisfied)).
Qed.

Lemma prove_unique_immutable_heap_block_fields_proof :
  prover_sound Provers.prove_unique_fully_constructed_immutable_heap_block
    well_formed_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ length simples = size
         ∧ eval_simples valuation simples = Some fields).
Proof.
  strictly (prove_unique_immutable_heap_block_fields_proof_in (m := Strict)).
Qed.

Lemma meet_block_field_simple_value_sound_in {m : mode}
    (min_name_mode : Name_mode.t)
    (field_kind : flambda_kind) (field_index : nat) :
  generic_value_sound_in m
    (Provers.meet_block_field_simple_value min_name_mode field_kind
       field_index)
    any_head (block_field_loads field_kind field_index)
    (fun statics valuation s v =>
       ∃ (tag : Z) (fields : list value) (field : value),
         block_of statics v = Some (tag, fields)
         ∧ nth_error fields field_index = Some field
         ∧ eval_simple valuation s = Some field).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead
    [tag [fields [field [Hblock [Hnth Hkind]]]]].
  assert (Hvariant : variants statics v).
  { right. exists tag, fields. exact Hblock. }
  pose proof (not_variant_head _ _ _ _ Hhead) as Hnot_variant.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    simpl; try exact I; try exact (Hnot_variant Hvariant).
  destruct blocks as [blocks | ]; [ | exact I ].
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead)
    as [[i [-> _]] | [tag' [fields' [Hblock' Hblocks]]]];
    [ discriminate Hblock | ].
  rewrite Hblock in Hblock'. injection Hblock' as <- <-.
  destruct (row_like_for_blocks_is_bottom blocks) eqn:Hbottom.
  - exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom Hblocks).
  - pose proof (get_field_satisfied _ _ _ _ _ field_index Hblocks) as Hfield.
    destruct (row_like_for_blocks_get_field blocks field_index)
      as [ | ty | ].
    + exact I.
    + destruct Hfield as [field' [Hnth' Hty]].
      rewrite Hnth in Hnth'. injection Hnth' as <-.
      destruct (kind_eqb (kind ty) field_kind) eqn:Hk; simpl.
      * destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
          as [[s | ] | ] eqn:Hcanonical;
          try exact I.
        exists tag, fields, field. split; [ exact Hblock | ].
        split; [ exact Hnth | ].
        exact
          (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hty
             Hcanonical).
      * pose proof (type_satisfied_kind _ _ _ _ Hty) as Hkind'.
        rewrite Hkind in Hkind'. rewrite Hkind' in Hk.
        rewrite kind_eqb_refl in Hk. discriminate Hk.
    + rewrite Hnth in Hfield. discriminate Hfield.
Qed.

Lemma meet_block_field_simple_value_sound (min_name_mode : Name_mode.t)
    (field_kind : flambda_kind) (field_index : nat) :
  generic_value_sound
    (Provers.meet_block_field_simple_value min_name_mode field_kind
       field_index)
    any_head (block_field_loads field_kind field_index)
    (fun statics valuation s v =>
       ∃ (tag : Z) (fields : list value) (field : value),
         block_of statics v = Some (tag, fields)
         ∧ nth_error fields field_index = Some field
         ∧ eval_simple valuation s = Some field).
Proof.
  strictly
    (meet_block_field_simple_value_sound_in (m := Strict) min_name_mode
       field_kind field_index).
Qed.

Lemma meet_block_field_simple_sound_proof_in {m : mode}
    (min_name_mode : Name_mode.t)
    (field_kind : flambda_kind) (field_index : nat) :
  meet_shortcut_sound_in m
    (fun tenv t =>
       Provers.meet_block_field_simple tenv min_name_mode field_kind t
         field_index)
    any_input (block_field_loads field_kind field_index)
    (fun statics valuation s v =>
       ∃ (tag : Z) (fields : list value) (field : value),
         block_of statics v = Some (tag, fields)
         ∧ nth_error fields field_index = Some field
         ∧ eval_simple valuation s = Some field).
Proof.
  apply
    (gen_value_to_meet_sound_in _ any_input any_head
       (block_field_loads field_kind field_index)).
  - exact any_input_to_head.
  - intros statics v Hloads.
    split; [ | split; [ | exact Hloads ] ];
      destruct Hloads as [tag [fields [field [Hblock _]]]].
    + exact (block_of_kind _ _ _ _ Hblock).
    + intros ->. discriminate Hblock.
  - exact (meet_block_field_simple_value_sound_in _ _ _).
Qed.

Lemma meet_block_field_simple_sound_proof (min_name_mode : Name_mode.t)
    (field_kind : flambda_kind) (field_index : nat) :
  meet_shortcut_sound
    (fun tenv t =>
       Provers.meet_block_field_simple tenv min_name_mode field_kind t
         field_index)
    any_input (block_field_loads field_kind field_index)
    (fun statics valuation s v =>
       ∃ (tag : Z) (fields : list value) (field : value),
         block_of statics v = Some (tag, fields)
         ∧ nth_error fields field_index = Some field
         ∧ eval_simple valuation s = Some field).
Proof.
  strictly
    (meet_block_field_simple_sound_proof_in (m := Strict) min_name_mode
       field_kind field_index).
Qed.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* A closure has, at its slot, the head of the closures of which
   nothing more is known. *)
Definition any_closure_at (slot : function_slot)
    : head_of_kind_value_non_null :=
  Closures
    (Row_like_for_closures
       [(slot,
         Row_like_case (Closures_entry [] [] [])
           (Row_like_index
              (Row_like_index_domain.At_least
                 (Set_of_closures_contents [] []))
              tt)
           empty_env_extension)])
    Alloc_mode.Heap.

Lemma any_closure_at_satisfied (statics : static_env) (vars : env)
    (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  closure_in_set statics v = Some (slot, functions, value_slots) ->
  head_of_kind_value_non_null_satisfied statics vars v (any_closure_at slot).
Proof.
  intros Hclosure. apply non_null_value_satisfied.
  apply
    (proj2
       (at_least_this_closure_satisfied statics vars v slot [] [] []
          Alloc_mode.Heap)).
  exists functions, value_slots. split; [ exact Hclosure | ].
  split; [ reflexivity | ]. split; [ reflexivity | ].
  apply closures_entry_lookup_iff.
  split; [ | split ]; intros; simpl in *; discriminate.
Qed.

(* A head of another form has no closure. *)
Lemma not_closures_head (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  head_of_kind_value_non_null_satisfied statics vars v head ->
  closures statics v ->
  match head with
  | Closures _ _ => True
  | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
    False
  end.
Proof.
  intros Hhead [slot [functions [value_slots Hclosure]]].
  pose proof
    (heads_compatible_satisfied _ _ _ _ _
       (any_closure_at_satisfied statics vars v slot functions value_slots
          Hclosure)
       Hhead)
    as Hcompatible.
  destruct head; exact Hcompatible.
Qed.

Lemma closures_domain (statics : static_env) (v : value) :
  closures statics v ->
  kind_of_value v = Value ∧ v <> Value_const Null ∧ closures statics v.
Proof.
  intros Hclosures. split; [ | split; [ | exact Hclosures ] ];
    destruct Hclosures as [slot [functions [value_slots Hclosure]]].
  - exact (closure_in_set_kind _ _ _ Hclosure).
  - intros ->. discriminate Hclosure.
Qed.

(* The code of a closure at a slot is that of the function type that
   an entry that holds of it has for the slot. *)
Lemma find_function_type_own_code (statics : static_env) (valuation : env)
    (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry)
    (exact : bool) (function_type : function_type) :
  closure_in_set statics v = Some (slot, functions, value_slots) ->
  closures_entry_satisfied statics valuation v slot functions value_slots
    entry ->
  closures_entry_find_function_type entry exact slot = Ok function_type ->
  lookup slot functions = Some (function_type_code_id function_type).
Proof.
  intros Hclosure Hentry Hfind.
  pose proof (closure_in_set_slot _ _ _ _ _ Hclosure) as Hbound.
  unfold is_bound in Hbound.
  destruct (lookup slot functions) as [code | ] eqn:Hcode;
    [ | discriminate Hbound ].
  rewrite
    (find_function_type_satisfied _ _ _ _ _ _ _ _ _ _ _ Hentry Hfind Hcode).
  reflexivity.
Qed.

Lemma single_closures_entry_of_singleton_sound (statics : static_env)
    (valuation : env) (v : value) (alloc_mode : Alloc_mode.t) (exact : bool)
    (slot : function_slot) (entry : closures_entry)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (x : function_slot * Alloc_mode.t * closures_entry * function_type) :
  closure_in_set statics v = Some (slot, functions, value_slots) ->
  closures_entry_satisfied statics valuation v slot functions value_slots
    entry ->
  Provers.single_closures_entry_of_singleton alloc_mode exact slot entry
  = Generic_proof.Proved x ->
  single_closures_entry_holds statics valuation x v.
Proof.
  intros Hclosure Hentry.
  unfold Provers.single_closures_entry_of_singleton.
  destruct (closures_entry_find_function_type entry exact slot)
    as [ | function_type | ] eqn:Hfind;
    try discriminate.
  intros [= <-]. exists functions, value_slots.
  split; [ exact Hclosure | ]. split; [ exact Hentry | ].
  exact (find_function_type_own_code _ _ _ _ _ _ _ _ _ Hclosure Hentry Hfind).
Qed.

(* An entry of a case that is not exact is never said to have no
   closure. *)
Lemma single_closures_entry_of_incomplete (alloc_mode : Alloc_mode.t)
    (slot : function_slot) (entry : closures_entry) :
  Provers.single_closures_entry_of_singleton alloc_mode false slot entry
  <> Generic_proof.Invalid.
Proof.
  unfold Provers.single_closures_entry_of_singleton.
  destruct entry as [function_types closure_types value_slot_types]. simpl.
  destruct (lookup slot function_types) as [[function_type | ] | ];
    discriminate.
Qed.

Lemma prove_single_closures_entry_generic_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_single_closures_entry_generic_value
    well_formed_head closures single_closures_entry_holds.
Proof.
  intros statics valuation tenv head v Hsat [_ Hwf] Hhead Hclosures.
  pose proof (not_closures_head _ _ _ _ Hhead Hclosures) as Hnot.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try exact Hnot.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [slot' [functions [value_slots [Hclosure Hrow]]]].
  simpl in Hwf.
  unfold Provers.prove_single_closures_entry_generic_value.
  destruct (row_like_for_closures_get_single_tag by_function_slot)
    as [[ | slot entry | slot entry] | ] eqn:Hsingle;
    simpl; try exact I.
  - destruct
      (get_single_tag_satisfied statics valuation v slot' functions
         value_slots by_function_slot slot entry)
      as [-> Hentry];
      [ rewrite Hsingle; reflexivity | exact Hrow | ].
    destruct
      (Provers.single_closures_entry_of_singleton alloc_mode true slot entry)
      as [x | | ] eqn:Hof.
    + exact
        (single_closures_entry_of_singleton_sound _ _ _ _ _ _ _ _ _ _
           Hclosure Hentry Hof).
    + exact I.
    + unfold Provers.single_closures_entry_of_singleton in Hof.
      destruct (closures_entry_find_function_type entry true slot)
        as [ | function_type | ] eqn:Hfind;
        try discriminate Hof.
      destruct by_function_slot as [known_closures].
      destruct (get_single_tag_exact known_closures slot entry Hsingle)
        as [contents [extension [Hlookup _]]].
      specialize (Hlookup slot). rewrite Pos.eqb_refl in Hlookup.
      apply (find_function_type_exact slot entry contents extension);
        [ | exact Hfind ].
      exact
        (proj2
           (proj1 (row_like_for_closures_well_formed_iff known_closures) Hwf)
           slot _ Hlookup).
  - destruct
      (get_single_tag_satisfied statics valuation v slot' functions
         value_slots by_function_slot slot entry)
      as [-> Hentry];
      [ rewrite Hsingle; reflexivity | exact Hrow | ].
    destruct
      (Provers.single_closures_entry_of_singleton alloc_mode false slot
         entry)
      as [x | | ] eqn:Hof.
    + exact
        (single_closures_entry_of_singleton_sound _ _ _ _ _ _ _ _ _ _
           Hclosure Hentry Hof).
    + exact I.
    + exact (single_closures_entry_of_incomplete _ _ _ Hof).
Qed.

Lemma prove_single_closures_entry_generic_value_sound :
  generic_value_sound Provers.prove_single_closures_entry_generic_value
    well_formed_head closures single_closures_entry_holds.
Proof.
  strictly (prove_single_closures_entry_generic_value_sound_in (m := Strict)).
Qed.

Lemma prove_single_closures_entry_generic_value_proved_sound_in {m : mode} :
  generic_value_proved_sound_in m
    Provers.prove_single_closures_entry_generic_value
    any_head single_closures_entry_holds.
Proof.
  intros statics valuation tenv head v x Hsat _ Hhead Hproved.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try discriminate Hproved.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [slot' [functions [value_slots [Hclosure Hrow]]]].
  unfold Provers.prove_single_closures_entry_generic_value in Hproved.
  destruct (row_like_for_closures_get_single_tag by_function_slot)
    as [[ | slot entry | slot entry] | ] eqn:Hsingle;
    simpl in Hproved; try discriminate Hproved;
    injection Hproved as Hproved;
    (destruct
       (get_single_tag_satisfied statics valuation v slot' functions
          value_slots by_function_slot slot entry)
       as [-> Hentry];
     [ rewrite Hsingle; reflexivity | exact Hrow | ]);
    exact
      (single_closures_entry_of_singleton_sound _ _ _ _ _ _ _ _ _ _
         Hclosure Hentry Hproved).
Qed.

Lemma prove_single_closures_entry_generic_value_proved_sound :
  generic_value_proved_sound Provers.prove_single_closures_entry_generic_value
    any_head single_closures_entry_holds.
Proof.
  strictly
    (prove_single_closures_entry_generic_value_proved_sound_in (m := Strict)).
Qed.

Lemma prove_single_closures_entry_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_single_closures_entry any_input
    single_closures_entry_holds.
Proof.
  apply (gen_value_to_proof_proved_sound_in _ any_input any_head).
  - exact any_input_to_head.
  - exact prove_single_closures_entry_generic_value_proved_sound_in.
Qed.

Lemma prove_single_closures_entry_sound_proof :
  prover_sound Provers.prove_single_closures_entry any_input
    single_closures_entry_holds.
Proof.
  strictly (prove_single_closures_entry_sound_proof_in (m := Strict)).
Qed.

Lemma meet_single_closures_entry_sound_proof_in {m : mode} :
  meet_shortcut_sound_in m Provers.meet_single_closures_entry well_formed_input
    closures single_closures_entry_holds.
Proof.
  apply
    (gen_value_to_meet_sound_in _ well_formed_input well_formed_head closures).
  - exact well_formed_input_to_head.
  - exact closures_domain.
  - exact prove_single_closures_entry_generic_value_sound_in.
Qed.

Lemma meet_single_closures_entry_sound_proof :
  meet_shortcut_sound Provers.meet_single_closures_entry well_formed_input
    closures single_closures_entry_holds.
Proof.
  strictly (meet_single_closures_entry_sound_proof_in (m := Strict)).
Qed.

(* The fold of [Provers.add_code_id_of_case]: once it is unknown it
   stays so; and if it ends known, every code id that it started with
   is in its answer, and so is that of the function type that each
   case gone through has for its own slot. *)
Lemma add_code_id_fold_unknown
    (cases : list (function_slot * row_like_closures_case)) :
  List.fold_left Provers.add_code_id_of_case cases Or_unknown.Unknown
  = Or_unknown.Unknown.
Proof.
  induction cases as [ | case cases IH]; [ reflexivity | exact IH ].
Qed.

Lemma add_code_id_fold
    (cases : list (function_slot * row_like_closures_case)) :
  ∀ acc code_ids : list code_id,
    List.fold_left Provers.add_code_id_of_case cases (Or_unknown.Known acc)
    = Or_unknown.Known code_ids ->
    (∀ code : code_id, code ∈ acc -> code ∈ code_ids)
    ∧ ∀ (slot : function_slot) (case : row_like_closures_case),
        (slot, case) ∈ cases ->
        ∃ function_type : function_type,
          closures_entry_find_function_type (case_maps_to case) false slot
          = Ok function_type
          ∧ function_type_code_id function_type ∈ code_ids.
Proof.
  induction cases as [ | [slot' case'] cases IH]; intros acc code_ids Hfold.
  - simpl in Hfold. injection Hfold as <-.
    split; [ auto | intros slot case [] ].
  - cbn [List.fold_left] in Hfold.
    destruct
      (Provers.add_code_id_of_case (Or_unknown.Known acc) (slot', case'))
      as [acc' | ] eqn:Hadd;
      [ | rewrite add_code_id_fold_unknown in Hfold; discriminate Hfold ].
    destruct (IH acc' code_ids Hfold) as [Hacc Hcases].
    unfold Provers.add_code_id_of_case in Hadd. simpl in Hadd.
    destruct (case_maps_to case')
      as [function_types closure_types value_slot_types] eqn:Hentry.
    destruct (lookup slot' function_types) as [[function_type | ] | ]
      eqn:Hlookup;
      try discriminate Hadd.
    injection Hadd as <-. split.
    + intros code Hin. apply Hacc. right. exact Hin.
    + intros slot case [Heq | Hin]; [ | exact (Hcases slot case Hin) ].
      injection Heq as <- <-. exists function_type. split.
      * rewrite Hentry. simpl. rewrite Hlookup. reflexivity.
      * apply Hacc. left. reflexivity.
Qed.

Lemma prove_code_ids_generic_value_sound_in {m : mode} :
  generic_value_sound_in m Provers.prove_code_ids_generic_value any_head
    closures
    (fun statics _ => closure_code_among statics).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead Hclosures.
  pose proof (not_closures_head _ _ _ _ Hhead Hclosures) as Hnot.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try exact Hnot.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [slot [functions [value_slots [Hclosure Hrow]]]].
  destruct by_function_slot as [known_closures].
  apply row_like_for_closures_lookup_iff in Hrow.
  destruct Hrow as [case [Hlookup [_ [Hentry _]]]].
  unfold Provers.prove_code_ids_generic_value.
  destruct
    (List.fold_left Provers.add_code_id_of_case (compact known_closures)
       (Or_unknown.Known []))
    as [code_ids | ] eqn:Hfold;
    [ | exact I ].
  destruct (add_code_id_fold _ _ _ Hfold) as [_ Hcases].
  destruct
    (Hcases slot case
       (proj2
          (keyed_in_compact Pos.eqb pos_eqb_decides_equality slot case
             known_closures)
          Hlookup))
    as [function_type [Hfind Hin]].
  pose proof
    (find_function_type_own_code _ _ _ _ _ _ _ _ _ Hclosure Hentry Hfind)
    as Hcode.
  destruct code_ids as [ | code_id code_ids]; simpl; [ destruct Hin | ].
  exists (function_type_code_id function_type), value_slots.
  split; [ | exact Hin ].
  apply closure_in_set_of. exists slot, functions.
  split; [ exact Hclosure | exact Hcode ].
Qed.

Lemma prove_code_ids_generic_value_sound :
  generic_value_sound Provers.prove_code_ids_generic_value any_head closures
    (fun statics _ => closure_code_among statics).
Proof.
  strictly (prove_code_ids_generic_value_sound_in (m := Strict)).
Qed.

Lemma prove_code_ids_sound_proof_in {m : mode} :
  prover_sound_in m Provers.prove_code_ids any_input
    (fun statics _ => closure_code_among statics).
Proof.
  intros statics valuation tenv t x v Hsat Hok Hin Hv Hproved.
  pose proof
    (gen_value_to_proof_silent Provers.prove_code_ids_generic_value) as _.
  unfold Provers.prove_code_ids, Provers.gen_value_to_proof in Hproved.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  destruct d as [[ | [[ | head | ] [ | is_null]] | ] | oub | k oub | oub
                | oub];
    try discriminate Hproved.
  pose proof (expanded_value_cases_in _ _ _ _ _ _ Hsat Hok Hv Hd) as Hcases.
  simpl in Hcases.
  destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
  assert (Hclosures : closures statics v).
  { destruct head as [is_int immediates get_tag blocks extensions is_unique
                     | alloc_mode | k contents alloc_mode
                     | by_function_slot alloc_mode | strs];
      try discriminate Hproved.
    rewrite closures_satisfied in Hhead.
    destruct Hhead as [slot [functions [value_slots [Hclosure _]]]].
    exists slot, functions, value_slots. exact Hclosure. }
  pose proof
    (prove_code_ids_generic_value_sound_in statics valuation tenv head v Hsat I
       Hhead Hclosures)
    as Hgen.
  destruct (Provers.prove_code_ids_generic_value tenv head) as [[y | | ] | ];
    try discriminate Hproved.
  injection Hproved as ->. exact Hgen.
Qed.

Lemma prove_code_ids_sound_proof :
  prover_sound Provers.prove_code_ids any_input
    (fun statics _ => closure_code_among statics).
Proof.
  strictly (prove_code_ids_sound_proof_in (m := Strict)).
Qed.

Lemma meet_code_ids_sound_proof_in {m : mode} :
  meet_shortcut_sound_in m Provers.meet_code_ids any_input closures
    (fun statics _ => closure_code_among statics).
Proof.
  apply (gen_value_to_meet_sound_in _ any_input any_head closures).
  - exact any_input_to_head.
  - exact closures_domain.
  - exact prove_code_ids_generic_value_sound_in.
Qed.

Lemma meet_code_ids_sound_proof :
  meet_shortcut_sound Provers.meet_code_ids any_input closures
    (fun statics _ => closure_code_among statics).
Proof.
  strictly (meet_code_ids_sound_proof_in (m := Strict)).
Qed.

Lemma meet_project_function_slot_simple_value_sound_in {m : mode}
    (min_name_mode : Name_mode.t) (move_to : function_slot) :
  generic_value_sound_in m
    (Provers.meet_project_function_slot_simple_value min_name_mode move_to)
    any_head (function_slot_projects move_to)
    (fun statics valuation s v =>
       ∀ (move_from : function_slot) (closure : value),
         project_function_slot statics move_from move_to v = Some closure ->
         eval_simple valuation s = Some closure).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead
    [move_from [closure Hproject]].
  destruct (project_function_slot_defined _ _ _ _ _ Hproject)
    as [functions [value_slots [code [code' [Hclosure _]]]]].
  assert (Hclosures : closures statics v).
  { exists move_from, functions, value_slots. exact Hclosure. }
  pose proof (not_closures_head _ _ _ _ Hhead Hclosures) as Hnot.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    try exact Hnot.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [slot [functions' [value_slots' [Hclosure' Hrow]]]].
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <- <-.
  unfold Provers.meet_project_function_slot_simple_value.
  destruct (row_like_for_closures_get_closure by_function_slot move_to)
    as [[ty | ] | ] eqn:Hget;
    simpl; try exact I.
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[s | ] | ] eqn:Hcanonical;
    try exact I.
  intros move_from' closure' Hproject'.
  destruct (project_function_slot_defined _ _ _ _ _ Hproject')
    as [functions' [value_slots' [code1 [code1' [Hclosure' _]]]]].
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <- <-.
  pose proof (get_closure_satisfied _ _ _ _ _ _ _ _ _ _ Hrow Hget Hproject')
    as Hty.
  exact (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hty Hcanonical).
Qed.

Lemma meet_project_function_slot_simple_value_sound
    (min_name_mode : Name_mode.t) (move_to : function_slot) :
  generic_value_sound
    (Provers.meet_project_function_slot_simple_value min_name_mode move_to)
    any_head (function_slot_projects move_to)
    (fun statics valuation s v =>
       ∀ (move_from : function_slot) (closure : value),
         project_function_slot statics move_from move_to v = Some closure ->
         eval_simple valuation s = Some closure).
Proof.
  strictly
    (meet_project_function_slot_simple_value_sound_in (m := Strict)
       min_name_mode move_to).
Qed.

Lemma meet_project_function_slot_simple_sound_proof_in {m : mode}
    (min_name_mode : Name_mode.t) (move_to : function_slot) :
  meet_shortcut_sound_in m
    (fun tenv t =>
       Provers.meet_project_function_slot_simple tenv min_name_mode t
         move_to)
    any_input (function_slot_projects move_to)
    (fun statics valuation s v =>
       ∀ (move_from : function_slot) (closure : value),
         project_function_slot statics move_from move_to v = Some closure ->
         eval_simple valuation s = Some closure).
Proof.
  apply
    (gen_value_to_meet_sound_in _ any_input any_head
       (function_slot_projects move_to)).
  - exact any_input_to_head.
  - intros statics v Hprojects.
    split; [ | split; [ | exact Hprojects ] ];
      destruct Hprojects as [move_from [closure Hproject]];
      destruct (project_function_slot_defined _ _ _ _ _ Hproject)
        as [functions [value_slots [code [code' [Hclosure _]]]]].
    + exact (closure_in_set_kind _ _ _ Hclosure).
    + intros ->. discriminate Hclosure.
  - exact (meet_project_function_slot_simple_value_sound_in _ _).
Qed.

Lemma meet_project_function_slot_simple_sound_proof
    (min_name_mode : Name_mode.t) (move_to : function_slot) :
  meet_shortcut_sound
    (fun tenv t =>
       Provers.meet_project_function_slot_simple tenv min_name_mode t
         move_to)
    any_input (function_slot_projects move_to)
    (fun statics valuation s v =>
       ∀ (move_from : function_slot) (closure : value),
         project_function_slot statics move_from move_to v = Some closure ->
         eval_simple valuation s = Some closure).
Proof.
  strictly
    (meet_project_function_slot_simple_sound_proof_in (m := Strict)
       min_name_mode move_to).
Qed.

Lemma meet_project_value_slot_simple_value_sound_in {m : mode}
    (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind) :
  generic_value_sound_in m
    (Provers.meet_project_value_slot_simple_value min_name_mode slot
       value_slot_kind)
    any_head (value_slot_projects value_slot_kind slot)
    (fun statics valuation s v =>
       ∀ (project_from : function_slot) (contents : value),
         project_value_slot statics project_from slot v = Some contents ->
         eval_simple valuation s = Some contents).
Proof.
  intros statics valuation tenv head v Hsat _ Hhead
    [project_from [contents [Hproject Hkind]]].
  destruct (proj1 (project_value_slot_defined _ _ _ _ _) Hproject)
    as [functions [value_slots [Hclosure Hcontents]]].
  assert (Hclosures : closures statics v).
  { exists project_from, functions, value_slots. exact Hclosure. }
  pose proof (not_closures_head _ _ _ _ Hhead Hclosures) as Hnot.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents' alloc_mode
                   | by_function_slot alloc_mode | strs];
    try exact Hnot.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [at_slot [functions' [value_slots' [Hclosure' Hrow]]]].
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <- <-.
  unfold Provers.meet_project_value_slot_simple_value.
  destruct (row_like_for_closures_get_env_var by_function_slot slot)
    as [[ty | ] | ] eqn:Hget;
    simpl; try exact I.
  pose proof (get_env_var_satisfied _ _ _ _ _ _ _ _ _ _ Hrow Hget Hcontents)
    as Hty.
  destruct (kind_eqb (kind ty) value_slot_kind) eqn:Hk; simpl.
  - destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
      as [[s | ] | ] eqn:Hcanonical;
      try exact I.
    intros project_from' contents' Hproject'.
    destruct (proj1 (project_value_slot_defined _ _ _ _ _) Hproject')
      as [functions' [value_slots' [Hclosure' Hcontents']]].
    rewrite Hclosure in Hclosure'. injection Hclosure' as <- <- <-.
    rewrite Hcontents in Hcontents'. injection Hcontents' as <-.
    exact
      (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hty Hcanonical).
  - pose proof (type_satisfied_kind _ _ _ _ Hty) as Hkind'.
    rewrite Hkind in Hkind'. rewrite Hkind' in Hk.
    rewrite kind_eqb_refl in Hk. discriminate Hk.
Qed.

Lemma meet_project_value_slot_simple_value_sound
    (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind) :
  generic_value_sound
    (Provers.meet_project_value_slot_simple_value min_name_mode slot
       value_slot_kind)
    any_head (value_slot_projects value_slot_kind slot)
    (fun statics valuation s v =>
       ∀ (project_from : function_slot) (contents : value),
         project_value_slot statics project_from slot v = Some contents ->
         eval_simple valuation s = Some contents).
Proof.
  strictly
    (meet_project_value_slot_simple_value_sound_in (m := Strict) min_name_mode
       slot value_slot_kind).
Qed.

Lemma meet_project_value_slot_simple_sound_proof_in {m : mode}
    (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind) :
  meet_shortcut_sound_in m
    (fun tenv t =>
       Provers.meet_project_value_slot_simple tenv min_name_mode t slot
         value_slot_kind)
    any_input (value_slot_projects value_slot_kind slot)
    (fun statics valuation s v =>
       ∀ (project_from : function_slot) (contents : value),
         project_value_slot statics project_from slot v = Some contents ->
         eval_simple valuation s = Some contents).
Proof.
  apply
    (gen_value_to_meet_sound_in _ any_input any_head
       (value_slot_projects value_slot_kind slot)).
  - exact any_input_to_head.
  - intros statics v Hprojects.
    split; [ | split; [ | exact Hprojects ] ];
      destruct Hprojects as [project_from [contents [Hproject _]]];
      destruct (proj1 (project_value_slot_defined _ _ _ _ _) Hproject)
        as [functions [value_slots [Hclosure _]]].
    + exact (closure_in_set_kind _ _ _ Hclosure).
    + intros ->. discriminate Hclosure.
  - exact (meet_project_value_slot_simple_value_sound_in _ _ _).
Qed.

Lemma meet_project_value_slot_simple_sound_proof
    (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind) :
  meet_shortcut_sound
    (fun tenv t =>
       Provers.meet_project_value_slot_simple tenv min_name_mode t slot
         value_slot_kind)
    any_input (value_slot_projects value_slot_kind slot)
    (fun statics valuation s v =>
       ∀ (project_from : function_slot) (contents : value),
         project_value_slot statics project_from slot v = Some contents ->
         eval_simple valuation s = Some contents).
Proof.
  strictly
    (meet_project_value_slot_simple_sound_proof_in (m := Strict) min_name_mode
       slot value_slot_kind).
Qed.

Lemma meet_project_value_slot_simple_known_proof_in {m : mode}
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (min_name_mode : Name_mode.t)
    (t : flambda_type) (slot : value_slot) (value_slot_kind : flambda_kind)
    (s : simple) (v : value) :
  sat m statics valuation tenv ->
  ok m v ->
  type_satisfied statics valuation v t ->
  Provers.meet_project_value_slot_simple tenv min_name_mode t slot
    value_slot_kind
  = Some (Meet_shortcut.Known_result s) ->
  ∀ (project_from : function_slot) (contents : value),
    project_value_slot statics project_from slot v = Some contents ->
    eval_simple valuation s = Some contents.
Proof.
  intros Hsat Hok Hv Hknown project_from contents Hproject.
  destruct (proj1 (project_value_slot_defined _ _ _ _ _) Hproject)
    as [functions [value_slots [Hclosure Hcontents]]].
  assert (Hnot_null : v <> Value_const Null).
  { intros ->. discriminate Hclosure. }
  revert project_from contents functions value_slots Hproject Hclosure
    Hcontents.
  refine
    (gen_value_to_meet_known_sound_in
       (Provers.meet_project_value_slot_simple_value min_name_mode slot
          value_slot_kind)
       (fun statics valuation s v =>
          ∀ (project_from : function_slot) (contents : value)
            (functions : list (function_slot * code_id))
            (value_slots : list (value_slot * value)),
            project_value_slot statics project_from slot v = Some contents ->
            closure_in_set statics v
            = Some (project_from, functions, value_slots) ->
            lookup slot value_slots = Some contents ->
            eval_simple valuation s = Some contents)
       statics valuation tenv t s v _ Hsat Hok Hv Hnot_null Hknown).
  intros head Hhead Hproved project_from contents functions value_slots _
    Hclosure Hcontents.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents' alloc_mode
                   | by_function_slot alloc_mode | strs];
    try discriminate Hproved.
  rewrite closures_satisfied in Hhead.
  destruct Hhead as [at_slot [functions' [value_slots' [Hclosure' Hrow]]]].
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <- <-.
  unfold Provers.meet_project_value_slot_simple_value in Hproved.
  destruct (row_like_for_closures_get_env_var by_function_slot slot)
    as [[ty | ] | ] eqn:Hget;
    simpl in Hproved; try discriminate Hproved.
  pose proof (get_env_var_satisfied _ _ _ _ _ _ _ _ _ _ Hrow Hget Hcontents)
    as Hty.
  destruct (kind_eqb (kind ty) value_slot_kind); simpl in Hproved;
    [ | discriminate Hproved ].
  destruct (Provers.canonical_simple_of_alias tenv min_name_mode ty)
    as [[s' | ] | ] eqn:Hcanonical;
    try discriminate Hproved.
  injection Hproved as <-.
  exact (canonical_simple_of_alias_sound_in _ _ _ _ _ _ _ Hsat Hty Hcanonical).
Qed.

Lemma meet_project_value_slot_simple_known_proof (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (t : flambda_type) (slot : value_slot) (value_slot_kind : flambda_kind)
    (s : simple) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t ->
  Provers.meet_project_value_slot_simple tenv min_name_mode t slot
    value_slot_kind
  = Some (Meet_shortcut.Known_result s) ->
  ∀ (project_from : function_slot) (contents : value),
    project_value_slot statics project_from slot v = Some contents ->
    eval_simple valuation s = Some contents.
Proof.
  intros Hsat.
  exact
    (meet_project_value_slot_simple_known_proof_in (m := Strict) statics
       valuation tenv min_name_mode t slot value_slot_kind s v Hsat I).
Qed.

(* ======================================================================== *)
(* Physical equality                                                        *)
(* ======================================================================== *)

Lemma in_inter_keyed {K : Type} (eqb : K -> K -> bool) (key : K)
    (keys keys' : list K) :
  decides_equality eqb ->
  key ∈ keys -> key ∈ keys' -> key ∈ Keyed.inter eqb keys keys'.
Proof.
  intros Heqb Hin Hin'.
  apply (keyed_among_in K eqb Heqb).
  rewrite (keyed_among_inter K eqb Heqb).
  rewrite (proj2 (keyed_among_in K eqb Heqb key keys) Hin).
  rewrite (proj2 (keyed_among_in K eqb Heqb key keys') Hin').
  reflexivity.
Qed.

(* Two sets with no member in common have no member in common. *)
Lemma disjoint_not_both {K : Type} (eqb : K -> K -> bool) (key : K)
    (keys keys' : list K) :
  decides_equality eqb ->
  Keyed.is_empty (Keyed.inter eqb keys keys') = true ->
  key ∈ keys -> key ∈ keys' -> False.
Proof.
  intros Heqb Hempty Hin Hin'.
  pose proof (in_inter_keyed eqb key keys keys' Heqb Hin Hin') as Hinter.
  rewrite (is_empty_nil _ Hempty) in Hinter. exact Hinter.
Qed.

(* One value has not two types of naked numbers that
   [Provers.incompatible_naked_numbers] tells apart. *)
Lemma incompatible_naked_numbers_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation v t1 ->
  type_satisfied statics valuation v t2 ->
  Provers.incompatible_naked_numbers tenv t1 t2 <> Some true.
Proof.
  intros Hsat Hv1 Hv2. unfold Provers.incompatible_naked_numbers.
  destruct (Provers.expand_head tenv t1) as [d1 | ] eqn:Hd1;
    [ | discriminate ].
  destruct (Provers.expand_head tenv t2) as [d2 | ] eqn:Hd2;
    [ | discriminate ].
  simpl.
  destruct d1 as [oub1 | oub1 | k1 [ | s1 | ] | oub1 | oub1];
    try discriminate;
    destruct d2 as [oub2 | oub2 | k2 [ | s2 | ] | oub2 | oub2];
    try discriminate.
  intros [= Hincompatible]. apply andb_prop in Hincompatible.
  destruct Hincompatible as [_ Hempty].
  pose proof (expanded_naked_number_cases _ _ _ _ _ _ _ Hsat Hv1 Hd1) as H1.
  pose proof (expanded_naked_number_cases _ _ _ _ _ _ _ Hsat Hv2 Hd2) as H2.
  cbv beta iota in H1, H2.
  destruct H1 as [c1 [n1 [-> [_ [Hn1 Hin1]]]]].
  destruct H2 as [c2 [n2 [Heq [_ [Hn2 Hin2]]]]].
  injection Heq as <-. rewrite Hn1 in Hn2. injection Hn2 as <-.
  exact
    (disjoint_not_both Z.eqb n1 s1 s2 z_eqb_decides_equality Hempty Hin1
       Hin2).
Qed.

(* What [Provers.physical_equality_of_immediates] says of a naked
   immediate of each type; and if it says that neither type has one,
   neither has. *)
Lemma physical_equality_of_immediates_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (imms1 imms2 : flambda_type)
    (proof : Generic_proof.t bool) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv imms1 ->
  well_formed_input tenv imms2 ->
  Provers.physical_equality_of_immediates tenv imms1 imms2 = Some proof ->
  (∀ i1 i2 : Z,
    type_satisfied statics valuation (Value_const (Naked_immediate i1))
      imms1 ->
    type_satisfied statics valuation (Value_const (Naked_immediate i2))
      imms2 ->
    match proof with
    | Generic_proof.Proved b => if b then i1 = i2 else i1 <> i2
    | Generic_proof.Invalid => False
    | Generic_proof.Unknown => True
    end)
  ∧ (proof = Generic_proof.Invalid ->
     ∀ i : Z,
       ~ type_satisfied statics valuation (Value_const (Naked_immediate i))
           imms1
       ∧ ~ type_satisfied statics valuation
             (Value_const (Naked_immediate i)) imms2).
Proof.
  intros Hsat Hwf1 Hwf2. unfold Provers.physical_equality_of_immediates.
  pose proof
    (fun i Hi =>
       prove_naked_immediates_generic_sound statics valuation tenv imms1
         (Value_const (Naked_immediate i)) Hsat Hwf1 Hi eq_refl)
    as Hsound1.
  pose proof
    (fun i Hi =>
       prove_naked_immediates_generic_sound statics valuation tenv imms2
         (Value_const (Naked_immediate i)) Hsat Hwf2 Hi eq_refl)
    as Hsound2.
  destruct (Provers.prove_naked_immediates_generic tenv imms1)
    as [proof1 | ];
    [ | discriminate ].
  destruct (Provers.prove_naked_immediates_generic tenv imms2)
    as [proof2 | ];
    [ | discriminate ].
  simpl.
  destruct proof1 as [set1 | | ], proof2 as [set2 | | ];
    cbv beta iota in Hsound1, Hsound2.
  - destruct (Keyed.is_empty (Keyed.inter Z.eqb set1 set2)) eqn:Hempty.
    + intros [= <-]. split; [ | discriminate ].
      intros i1 i2 H1 H2 Heq. subst i2.
      destruct (Hsound1 _ H1) as [i [[= <-] Hin1]].
      destruct (Hsound2 _ H2) as [i [[= <-] Hin2]].
      exact
        (disjoint_not_both Z.eqb i1 set1 set2 z_eqb_decides_equality Hempty
           Hin1 Hin2).
    + destruct (Keyed.get_singleton Z.eqb set1) as [imm1 | ] eqn:Hsingle1;
        [ | intros [= <-]; split; [ intros; exact I | discriminate ] ].
      destruct (Keyed.get_singleton Z.eqb set2) as [imm2 | ] eqn:Hsingle2;
        [ | intros [= <-]; split; [ intros; exact I | discriminate ] ].
      destruct (Z.eqb imm1 imm2) eqn:Hsame; [ | discriminate ].
      intros [= <-]. split; [ | discriminate ].
      intros i1 i2 H1 H2.
      destruct (Hsound1 _ H1) as [i [[= <-] Hin1]].
      destruct (Hsound2 _ H2) as [i [[= <-] Hin2]].
      rewrite (get_singleton_in _ _ _ Hsingle1 Hin1).
      rewrite (get_singleton_in _ _ _ Hsingle2 Hin2).
      exact (proj1 (Z.eqb_eq _ _) Hsame).
  - intros [= <-]. split; [ intros; exact I | discriminate ].
  - intros [= <-]. split; [ | discriminate ].
    intros i1 i2 H1 H2. destruct (Hsound2 _ H2).
  - intros [= <-]. split; [ intros; exact I | discriminate ].
  - intros [= <-]. split; [ intros; exact I | discriminate ].
  - intros [= <-]. split; [ | discriminate ].
    intros i1 i2 H1 H2. destruct (Hsound2 _ H2).
  - intros [= <-]. split; [ | discriminate ].
    intros i1 i2 H1 H2. destruct (Hsound1 _ H1).
  - intros [= <-]. split; [ | discriminate ].
    intros i1 i2 H1 H2. destruct (Hsound1 _ H1).
  - intros [= <-]. split.
    + intros i1 i2 H1 H2. exact (Hsound1 _ H1).
    + intros _ i. split; intros Hi; [ exact (Hsound1 _ Hi) | ].
      exact (Hsound2 _ Hi).
Qed.

(* The shape that [get_singleton] answers of a well-formed structure
   is written in one way. *)
Lemma get_singleton_canonical (blocks : row_like_for_blocks) (tag : Z)
    (shape : block_shape) (size : nat) (tys : list flambda_type)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_well_formed blocks = true ->
  row_like_for_blocks_get_singleton blocks
  = Some (tag, shape, size, tys, alloc_mode) ->
  block_shape_canonical shape.
Proof.
  intros Hwf Hsingleton.
  destruct blocks as [known_tags other_tags alloc_mode'].
  destruct (get_singleton_case _ _ _ _ _ _ _ _ Hsingleton)
    as [_ [case [Hlookup [_ [Hshape _]]]]].
  specialize (Hlookup tag). rewrite Z.eqb_refl in Hlookup.
  pose proof
    (proj1
       (proj2
          (proj1
             (row_like_for_blocks_well_formed_iff known_tags other_tags
                alloc_mode')
             Hwf))
       tag case Hlookup)
    as Hcase.
  apply row_like_block_case_well_formed_iff in Hcase.
  destruct Hcase as [_ [Hcanonical _]].
  rewrite <- Hshape. exact Hcanonical.
Qed.

(* What [Provers.physical_equality_of_blocks] answers: unknown, if a
   block is of both structures; that neither has a block, only if
   neither has; and never that two blocks are the same. *)
Lemma physical_equality_of_blocks_sound (statics : static_env)
    (valuation : env) (blocks1 blocks2 : row_like_for_blocks) :
  row_like_for_blocks_well_formed blocks1 = true ->
  row_like_for_blocks_well_formed blocks2 = true ->
  (∀ (tag : Z) (fields : list value),
    row_like_for_blocks_satisfied statics valuation tag fields blocks1 ->
    row_like_for_blocks_satisfied statics valuation tag fields blocks2 ->
    Provers.physical_equality_of_blocks blocks1 blocks2
    = Generic_proof.Unknown)
  ∧ (Provers.physical_equality_of_blocks blocks1 blocks2
     = Generic_proof.Invalid ->
     ∀ (tag : Z) (fields : list value),
       ~ row_like_for_blocks_satisfied statics valuation tag fields blocks1
       ∧ ~ row_like_for_blocks_satisfied statics valuation tag fields
             blocks2)
  ∧ Provers.physical_equality_of_blocks blocks1 blocks2
    <> Generic_proof.Proved true.
Proof.
  intros Hwf1 Hwf2. unfold Provers.physical_equality_of_blocks.
  destruct (row_like_for_blocks_is_bottom blocks1) eqn:Hbottom1;
    destruct (row_like_for_blocks_is_bottom blocks2) eqn:Hbottom2;
    simpl.
  - split; [ | split; [ | discriminate ] ].
    + intros tag fields H1 _. exfalso.
      exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom1 H1).
    + intros _ tag fields.
      split;
        [ exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom1)
        | exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom2)
        ].
  - split; [ | split; discriminate ].
    intros tag fields H1 _. exfalso.
    exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom1 H1).
  - split; [ | split; discriminate ].
    intros tag fields _ H2. exfalso.
    exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom2 H2).
  - destruct (row_like_for_blocks_get_singleton blocks1)
      as [[[[[tag1 shape1] size1] tys1] alloc_mode1] | ] eqn:Hsingle1;
      [ | split; [ intros; reflexivity | split; discriminate ] ].
    destruct (row_like_for_blocks_get_singleton blocks2)
      as [[[[[tag2 shape2] size2] tys2] alloc_mode2] | ] eqn:Hsingle2;
      [ | split; [ intros; reflexivity | split; discriminate ] ].
    destruct
      (Z.eqb tag1 tag2 && Nat.eqb size1 size2
       && block_shape_eqb shape1 shape2)
      eqn:Hsame;
      [ split; [ intros; reflexivity | split; discriminate ] | ].
    split; [ | split; discriminate ].
    intros tag fields H1 H2. exfalso.
    destruct (get_singleton_satisfied _ _ _ _ _ _ _ _ _ _ Hsingle1 H1)
      as [-> [Hlength1 [Hshape1 _]]].
    destruct (get_singleton_satisfied _ _ _ _ _ _ _ _ _ _ Hsingle2 H2)
      as [Htag [Hlength2 [Hshape2 _]]].
    subst tag2.
    pose proof
      (block_shapes_disjoint shape1 shape2 tag1 fields
         (get_singleton_canonical _ _ _ _ _ _ Hwf1 Hsingle1)
         (get_singleton_canonical _ _ _ _ _ _ Hwf2 Hsingle2) Hshape1 Hshape2)
      as Hshape.
    subst shape2. rewrite <- Hlength1, <- Hlength2 in Hsame.
    rewrite Z.eqb_refl, Nat.eqb_refl in Hsame.
    rewrite (proj2 (block_shape_eqb_eq shape1 shape1) eq_refl) in Hsame.
    discriminate Hsame.
Qed.

(* The blocks of a well-formed variant are well formed. *)
Lemma well_formed_head_blocks (tenv : Typing_env.t)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  well_formed_head tenv
    (Variant is_int immediates get_tag (Or_unknown.Known blocks) extensions
       is_unique) ->
  row_like_for_blocks_well_formed blocks = true.
Proof.
  intros [_ Hwf]. rewrite variant_well_formed in Hwf.
  apply andb_prop in Hwf. destruct Hwf as [Hwf _].
  apply andb_prop in Hwf. exact (proj2 Hwf).
Qed.

(* A variant with no immediate is not a variant with no block. *)
Lemma no_immediate_and_no_block_sound (statics : static_env)
    (valuation : env) (v1 v2 : value) (is_int1 is_int2 : option variable)
    (immediates1 immediates2 : Or_unknown.t flambda_type)
    (get_tag1 get_tag2 : option variable)
    (blocks1 blocks2 : Or_unknown.t row_like_for_blocks)
    (extensions1 extensions2 : variant_extensions)
    (is_unique1 is_unique2 : bool) :
  head_of_kind_value_non_null_satisfied statics valuation v1
    (Variant is_int1 immediates1 get_tag1 blocks1 extensions1 is_unique1) ->
  head_of_kind_value_non_null_satisfied statics valuation v2
    (Variant is_int2 immediates2 get_tag2 blocks2 extensions2 is_unique2) ->
  Provers.no_immediate_and_no_block immediates1 blocks2 = true ->
  v1 <> v2.
Proof.
  intros Hhead1 Hhead2 Htest Heq. subst v2.
  unfold Provers.no_immediate_and_no_block in Htest.
  destruct immediates1 as [imms1 | ]; [ | discriminate Htest ].
  destruct blocks2 as [blocks2 | ]; [ | discriminate Htest ].
  apply andb_prop in Htest. destruct Htest as [Himms Hblocks].
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead1)
    as [[i [-> Hi]] | [tag [fields [Hblock _]]]].
  - exact (is_obviously_bottom_satisfied _ _ _ _ Himms Hi).
  - destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead2)
      as [[i [-> _]] | [tag' [fields' [Hblock' Hrow]]]].
    + discriminate Hblock.
    + exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hblocks Hrow).
Qed.

(* How the answers for the immediates and for the blocks of two
   variants make a proof. *)
Lemma physical_equality_of_parts_cases
    (immediates_equality blocks_equality : Generic_proof.t bool) (b : bool) :
  Provers.physical_equality_of_parts immediates_equality blocks_equality
  = Proof_of_property.Proved b ->
  (blocks_equality = Generic_proof.Invalid
   ∧ immediates_equality = Generic_proof.Proved b)
  ∨ (immediates_equality = Generic_proof.Invalid
     ∧ blocks_equality = Generic_proof.Proved b)
  ∨ (b = false
     ∧ immediates_equality = Generic_proof.Proved false
     ∧ blocks_equality = Generic_proof.Proved false).
Proof.
  destruct immediates_equality as [[ | ] | | ],
    blocks_equality as [[ | ] | | ];
    simpl; try discriminate; intros [= <-]; auto.
Qed.

Lemma physical_equality_of_variants_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (v1 v2 : value)
    (is_int1 is_int2 : option variable)
    (immediates1 immediates2 : Or_unknown.t flambda_type)
    (get_tag1 get_tag2 : option variable)
    (blocks1 blocks2 : Or_unknown.t row_like_for_blocks)
    (extensions1 extensions2 : variant_extensions)
    (is_unique1 is_unique2 : bool) (b : bool) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_head tenv
    (Variant is_int1 immediates1 get_tag1 blocks1 extensions1 is_unique1) ->
  well_formed_head tenv
    (Variant is_int2 immediates2 get_tag2 blocks2 extensions2 is_unique2) ->
  head_of_kind_value_non_null_satisfied statics valuation v1
    (Variant is_int1 immediates1 get_tag1 blocks1 extensions1 is_unique1) ->
  head_of_kind_value_non_null_satisfied statics valuation v2
    (Variant is_int2 immediates2 get_tag2 blocks2 extensions2 is_unique2) ->
  Provers.physical_equality_of_variants tenv immediates1 immediates2 blocks1
    blocks2
  = Some (Proof_of_property.Proved b) ->
  if b then v1 = v2 else v1 <> v2.
Proof.
  intros Hsat Hwf1 Hwf2 Hhead1 Hhead2.
  unfold Provers.physical_equality_of_variants.
  destruct (Provers.no_immediate_and_no_block immediates1 blocks2)
    eqn:Hguard1.
  { intros [= <-].
    exact
      (no_immediate_and_no_block_sound _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
         Hhead1 Hhead2 Hguard1). }
  destruct (Provers.no_immediate_and_no_block immediates2 blocks1)
    eqn:Hguard2.
  { intros [= <-] Heq.
    exact
      (no_immediate_and_no_block_sound _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
         Hhead2 Hhead1 Hguard2 (eq_sym Heq)). }
  destruct immediates1 as [imms1 | ]; [ | discriminate ].
  destruct immediates2 as [imms2 | ]; [ | discriminate ].
  destruct blocks1 as [blocks1 | ]; [ | discriminate ].
  destruct blocks2 as [blocks2 | ]; [ | discriminate ].
  destruct (Provers.physical_equality_of_immediates tenv imms1 imms2)
    as [immediates_equality | ] eqn:Himmediates;
    [ | discriminate ].
  simpl. intros [= Hresult].
  destruct
    (physical_equality_of_immediates_sound _ _ _ _ _ _ Hsat
       (well_formed_head_immediates _ _ _ _ _ _ _ Hwf1)
       (well_formed_head_immediates _ _ _ _ _ _ _ Hwf2) Himmediates)
    as [Himms_values Himms_invalid].
  destruct
    (physical_equality_of_blocks_sound statics valuation blocks1 blocks2
       (well_formed_head_blocks _ _ _ _ _ _ _ Hwf1)
       (well_formed_head_blocks _ _ _ _ _ _ _ Hwf2))
    as [Hblocks_common [Hblocks_invalid Hblocks_true]].
  destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead1)
    as [[i1 [-> Hi1]] | [tag1 [fields1 [Hblock1 Hrow1]]]];
    destruct (variant_cases _ _ _ _ _ _ _ _ _ Hhead2)
      as [[i2 [-> Hi2]] | [tag2 [fields2 [Hblock2 Hrow2]]]];
    destruct (physical_equality_of_parts_cases _ _ _ Hresult)
      as [[Hblocks Himms] | [[Himms Hblocks] | [-> [Himms Hblocks]]]].
  - specialize (Himms_values i1 i2 Hi1 Hi2). rewrite Himms in Himms_values.
    destruct b; [ rewrite Himms_values; reflexivity | ].
    intros [= Heq]. exact (Himms_values Heq).
  - destruct (proj1 (Himms_invalid Himms i1) Hi1).
  - specialize (Himms_values i1 i2 Hi1 Hi2). rewrite Himms in Himms_values.
    intros [= Heq]. exact (Himms_values Heq).
  - destruct (proj2 (Hblocks_invalid Hblocks tag2 fields2) Hrow2).
  - destruct (proj1 (Himms_invalid Himms i1) Hi1).
  - intros <-. discriminate Hblock2.
  - destruct (proj1 (Hblocks_invalid Hblocks tag1 fields1) Hrow1).
  - destruct (proj2 (Himms_invalid Himms i2) Hi2).
  - intros ->. discriminate Hblock1.
  - destruct (proj1 (Hblocks_invalid Hblocks tag1 fields1) Hrow1).
  - destruct b; [ contradiction | ].
    intros <-. rewrite Hblock1 in Hblock2. injection Hblock2 as <- <-.
    rewrite (Hblocks_common tag1 fields1 Hrow1 Hrow2) in Hblocks.
    discriminate Hblocks.
  - intros <-. rewrite Hblock1 in Hblock2. injection Hblock2 as <- <-.
    rewrite (Hblocks_common tag1 fields1 Hrow1 Hrow2) in Hblocks.
    discriminate Hblocks.
Qed.

Lemma physical_equality_of_heads_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t)
    (head1 head2 : head_of_kind_value_non_null) (v1 v2 : value) (b : bool) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_head tenv head1 ->
  well_formed_head tenv head2 ->
  head_of_kind_value_non_null_satisfied statics valuation v1 head1 ->
  head_of_kind_value_non_null_satisfied statics valuation v2 head2 ->
  Provers.physical_equality_of_heads tenv head1 head2
  = Some (Proof_of_property.Proved b) ->
  if b then v1 = v2 else v1 <> v2.
Proof.
  intros Hsat Hwf1 Hwf2 Hhead1 Hhead2 Hresult.
  assert (Hcompatible : v1 = v2 -> heads_compatible head1 head2).
  { intros <-. exact (heads_compatible_satisfied _ _ _ _ _ Hhead1 Hhead2). }
  unfold Provers.physical_equality_of_heads in Hresult.
  destruct head1 as [is_int1 immediates1 get_tag1 blocks1 extensions1
                       is_unique1
                    | alloc_mode1 | k1 contents1 alloc_mode1
                    | by_function_slot1 alloc_mode1 | strs1];
    try (rewrite mutable_block_head_satisfied in Hhead1; destruct Hhead1);
    destruct head2 as [is_int2 immediates2 get_tag2 blocks2 extensions2
                         is_unique2
                      | alloc_mode2 | k2 contents2 alloc_mode2
                      | by_function_slot2 alloc_mode2 | strs2];
    try (rewrite mutable_block_head_satisfied in Hhead2; destruct Hhead2);
    cbv beta iota in Hresult; unfold heads_compatible in Hcompatible;
    try discriminate Hresult;
    try (injection Hresult as <-; exact Hcompatible).
  - exact
      (physical_equality_of_variants_sound _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
         _ _ Hsat Hwf1 Hwf2 Hhead1 Hhead2 Hresult).
  - destruct (boxable_number_eqb k1 k2) eqn:Hk.
    + destruct
        (Provers.incompatible_naked_numbers tenv contents1 contents2)
        as [[ | ] | ] eqn:Hincompatible;
        try discriminate Hresult.
      injection Hresult as <-. intros Heq. subst v2.
      rewrite boxed_number_satisfied in Hhead1, Hhead2.
      destruct Hhead1 as [c1 [Hboxed1 [_ Hc1]]].
      destruct Hhead2 as [c2 [Hboxed2 [_ Hc2]]].
      rewrite Hboxed1 in Hboxed2. injection Hboxed2 as <-.
      exact
        (incompatible_naked_numbers_sound _ _ _ _ _ _ Hsat Hc1 Hc2
           Hincompatible).
    + injection Hresult as <-. intros Heq.
      rewrite (Hcompatible Heq) in Hk.
      rewrite (proj2 (boxable_number_eqb_eq k2 k2) eq_refl) in Hk.
      discriminate Hk.
  - destruct (Keyed.is_empty (Keyed.inter (lists_eqb Z.eqb) strs1 strs2))
      eqn:Hempty;
      [ | discriminate Hresult ].
    injection Hresult as <-. intros Heq. subst v2.
    rewrite string_satisfied in Hhead1, Hhead2.
    destruct Hhead1 as [bytes1 [Hstring1 Hin1]].
    destruct Hhead2 as [bytes2 [Hstring2 Hin2]].
    rewrite Hstring1 in Hstring2. injection Hstring2 as <-.
    exact
      (disjoint_not_both (lists_eqb Z.eqb) bytes1 strs1 strs2 bytes_eqb_eq
         Hempty Hin1 Hin2).
Qed.

Lemma physical_equality_check_heads_sound (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (v1 v2 : value) (b : bool) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv t1 ->
  well_formed_input tenv t2 ->
  type_satisfied statics valuation v1 t1 ->
  type_satisfied statics valuation v2 t2 ->
  Provers.physical_equality_check_heads tenv t1 t2
  = Some (Proof_of_property.Proved b) ->
  if b then v1 = v2 else v1 <> v2.
Proof.
  intros Hsat Hin1 Hin2 Hv1 Hv2.
  unfold Provers.physical_equality_check_heads.
  destruct (Provers.expand_head tenv t1) as [d1 | ] eqn:Hd1;
    [ | discriminate ].
  destruct (Provers.expand_head tenv t2) as [d2 | ] eqn:Hd2;
    [ | discriminate ].
  simpl.
  destruct d1 as [[ | [non_null1 is_null1] | ] | oub1 | k1 oub1 | oub1
                 | oub1];
    try discriminate;
    destruct d2 as [[ | [non_null2 is_null2] | ] | oub2 | k2 oub2 | oub2
                   | oub2];
    try discriminate.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv1 Hd1) as Hcases1.
  pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv2 Hd2) as Hcases2.
  cbv beta iota in Hcases1, Hcases2.
  destruct is_null1 as [ | x1], is_null2 as [ | x2]; try discriminate.
  - destruct non_null1 as [ | head1 | ]; try discriminate;
      destruct non_null2 as [ | head2 | ]; try discriminate.
    intros Hresult.
    destruct Hcases1 as [[_ Hnot] | [_ Hhead1]]; [ contradiction | ].
    destruct Hcases2 as [[_ Hnot] | [_ Hhead2]]; [ contradiction | ].
    exact
      (physical_equality_of_heads_sound _ _ _ _ _ _ _ _ Hsat
         (well_formed_input_to_head _ _ _ _ Hin1 Hd1)
         (well_formed_input_to_head _ _ _ _ Hin2 Hd2) Hhead1 Hhead2 Hresult).
  - destruct non_null1 as [ | head1 | ]; try discriminate;
      destruct non_null2 as [ | head2 | ]; try discriminate.
    intros [= <-].
    destruct Hcases1 as [[-> _] | [_ []]].
    destruct Hcases2 as [[-> _] | [_ []]]. reflexivity.
Qed.

Lemma prove_physical_equality_sound_proof (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (b : bool) (v1 v2 : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv t1 ->
  well_formed_input tenv t2 ->
  type_satisfied statics valuation v1 t1 ->
  type_satisfied statics valuation v2 t2 ->
  Provers.prove_physical_equality tenv t1 t2
  = Some (Proof_of_property.Proved b) ->
  if b then v1 = v2 else v1 <> v2.
Proof.
  intros Hsat Hin1 Hin2 Hv1 Hv2.
  pose proof
    (physical_equality_check_heads_sound _ _ _ _ _ _ _ b Hsat Hin1 Hin2 Hv1
       Hv2)
    as Hcheck.
  unfold Provers.prove_physical_equality.
  destruct (get_alias_opt t1) as [s1 | ] eqn:Halias1;
    [ | destruct (get_alias_opt t2); exact Hcheck ].
  destruct (get_alias_opt t2) as [s2 | ] eqn:Halias2; [ | exact Hcheck ].
  pose proof (proj1 (proj1 (get_alias_satisfied _ _ _ _ _ Halias1) Hv1))
    as Hs1.
  pose proof (proj1 (proj1 (get_alias_satisfied _ _ _ _ _ Halias2) Hv2))
    as Hs2.
  destruct s1 as [x1 | sym1 | c1], s2 as [x2 | sym2 | c2];
    try exact Hcheck; simpl in Hs1, Hs2.
  - destruct (Pos.eqb_spec x1 x2) as [-> | Hne]; [ | exact Hcheck ].
    intros [= <-]. rewrite Hs1 in Hs2. injection Hs2 as <-. reflexivity.
  - injection Hs1 as <-. injection Hs2 as <-.
    destruct (Pos.eqb_spec sym1 sym2) as [-> | Hne]; intros [= <-];
      [ reflexivity | ].
    intros [= Heq]. exact (Hne Heq).
  - injection Hs1 as <-. injection Hs2 as <-. intros [= <-]. discriminate.
  - injection Hs1 as <-. injection Hs2 as <-. intros [= <-]. discriminate.
  - injection Hs1 as <-. injection Hs2 as <-. unfold const_eqb.
    destruct (const_eq_dec c1 c2) as [-> | Hne]; intros [= <-];
      [ reflexivity | ].
    intros [= Heq]. exact (Hne Heq).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem prove_is_int_sound : prove_is_int_sound_statement.
Proof. exact prove_is_int_sound_proof. Qed.

Theorem meet_is_int_variant_only_sound
  : meet_is_int_variant_only_sound_statement.
Proof. exact meet_is_int_variant_only_sound_proof. Qed.

Theorem prove_is_not_a_pointer_sound : prove_is_not_a_pointer_sound_statement.
Proof. exact prove_is_not_a_pointer_sound_proof. Qed.

Theorem prove_get_tag_sound : prove_get_tag_sound_statement.
Proof. exact prove_get_tag_sound_proof. Qed.

Theorem meet_is_null_sound : meet_is_null_sound_statement.
Proof. exact meet_is_null_sound_proof. Qed.

Theorem meet_naked_immediates_sound : meet_naked_immediates_sound_statement.
Proof. exact meet_naked_immediates_sound_proof. Qed.

Theorem prove_equals_tagged_immediates_sound
  : prove_equals_tagged_immediates_sound_statement.
Proof. exact prove_equals_tagged_immediates_sound_proof. Qed.

Theorem meet_equals_tagged_immediates_sound
  : meet_equals_tagged_immediates_sound_statement.
Proof. exact meet_equals_tagged_immediates_sound_proof. Qed.

Theorem meet_equals_single_tagged_immediate_sound
  : meet_equals_single_tagged_immediate_sound_statement.
Proof. exact meet_equals_single_tagged_immediate_sound_proof. Qed.

Theorem meet_naked_number_sound : meet_naked_number_sound_statement.
Proof. exact meet_naked_number_sound_proof. Qed.

Theorem provers_silent_on_poison : provers_silent_on_poison_statement.
Proof.
  repeat split.
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
  - exact (prove_is_not_a_pointer_silent).
  - exact (gen_value_to_proof_silent _).
  - exact (meet_naked_immediates_silent).
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
  - exact (meet_equals_single_tagged_immediate_silent).
  - exact (meet_naked_number_silent).
Qed.

Theorem meet_is_null_poison : meet_is_null_poison_statement.
Proof. exact meet_is_null_poison_proof. Qed.

(* ======================================================================== *)
(* The answers against the primitives                                       *)
(* ======================================================================== *)

Lemma tagged_immediate_of_value (v : value) (i : Z) :
  int_of_value Standard_tagged_immediate v = Some i ->
  v = Value_const (Tagged_immediate i).
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c
       | l | callee args args_arity];
    try discriminate.
  destruct c; try discriminate. simpl.
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test
  end;
    [ intros [= <-]; reflexivity | discriminate ].
Qed.

Theorem is_int_defined : is_int_defined_statement.
Proof.
  intros statics variant_only v r. rewrite eval_prim_unary. simpl.
  unfold eval_is_int.
  destruct (is_tagged_immediate v) eqn:Htagged.
  - simpl. intros [= <-].
    pose proof (tagged_immediate_is_immediate _ Htagged) as Himmediate.
    split; [ intros _; left; exact Himmediate | ].
    exists true. split; [ reflexivity | exact Himmediate ].
  - destruct variant_only.
    + destruct (block_of statics v) as [[tag fields] | ] eqn:Hblock;
        [ | discriminate ].
      simpl. intros [= <-].
      split; [ intros _; right; exists tag, fields; exact Hblock | ].
      exists false. split; [ reflexivity | ].
      intros Himmediate.
      rewrite (immediate_not_block statics v Himmediate) in Hblock.
      discriminate Hblock.
    + destruct (is_allocated statics v) eqn:Hallocated; [ | discriminate ].
      simpl. intros [= <-].
      split; [ discriminate | ].
      exists false. split; [ reflexivity | ].
      intros [i ->]. discriminate Hallocated.
Qed.

Lemma eval_is_null_cases (statics : static_env) (v : value) (b : bool) :
  eval_is_null statics v = Some b ->
  kind_of_value v = Value ∧ is_null_is b v.
Proof.
  intros H.
  assert (Hnull : v = Value_const Null ∨ v <> Value_const Null).
  { destruct v as [c | | | | | | | | ]; [ destruct c | | | | | | | | ];
      first [ left; reflexivity | right; discriminate ]. }
  destruct Hnull as [-> | Hnot_null].
  - unfold eval_is_null in H. injection H as <-. split; reflexivity.
  - assert (Heq :
      eval_is_null statics v
      = if is_tagged_immediate v || is_allocated statics v
        then Some false
        else None).
    { unfold eval_is_null.
      destruct v as [c | | | | | | | | ]; [ destruct c | | | | | | | | ];
        first [ reflexivity | exfalso; apply Hnot_null; reflexivity ]. }
    rewrite Heq in H.
    destruct (is_tagged_immediate v) eqn:Htagged; simpl in H.
    + injection H as <-.
      split; [ exact (is_tagged_immediate_kind _ Htagged) | exact Hnot_null ].
    + destruct (is_allocated statics v) eqn:Hallocated; [ | discriminate H ].
      injection H as <-.
      split; [ | exact Hnot_null ].
      exact (proj1 (allocated_not_null _ _ Hallocated)).
Qed.

Theorem is_null_defined : is_null_defined_statement.
Proof.
  intros statics v r. rewrite eval_prim_unary. simpl.
  destruct (eval_is_null statics v) as [b | ] eqn:Hb; [ | discriminate ].
  simpl. intros [= <-].
  destruct (eval_is_null_cases _ _ _ Hb) as [Hkind His_null].
  split; [ exact Hkind | ].
  exists b. split; [ reflexivity | exact His_null ].
Qed.

Theorem get_tag_defined : get_tag_defined_statement.
Proof.
  intros statics v r. rewrite eval_prim_unary. simpl.
  destruct (block_of statics v) as [[tag fields] | ]; [ | discriminate ].
  simpl. intros [= <-]. exists tag, fields. split; reflexivity.
Qed.

Theorem untag_immediate_defined : untag_immediate_defined_statement.
Proof.
  intros statics v r. rewrite eval_prim_unary.
  change (eval_unary_primitive statics Untag_immediate v)
    with
      (let* i := int_of_value Standard_tagged_immediate v in
       Some (Value_const (Naked_immediate i))).
  destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hint;
    [ | discriminate ].
  intros [= <-]. exists i.
  split; [ exact (tagged_immediate_of_value _ _ Hint) | reflexivity ].
Qed.

Lemma is_int_is_unique (b b' : bool) (v : value) :
  is_int_is b v -> is_int_is b' v -> b = b'.
Proof.
  destruct b, b'; simpl; intros H H'; first [ reflexivity | contradiction ].
Qed.

Theorem prove_is_int_is_int : prove_is_int_is_int_statement.
Proof.
  intros statics valuation tenv t b v r Hsat Hv Hproved Heval.
  pose proof (prove_is_int_sound _ _ _ _ _ _ Hsat I Hv Hproved)
    as Hb.
  destruct (is_int_defined _ _ _ _ Heval) as [_ [b' [-> Hb']]].
  rewrite (is_int_is_unique _ _ _ Hb Hb'). reflexivity.
Qed.

Theorem prove_is_int_get_tag : prove_is_int_get_tag_statement.
Proof.
  intros statics valuation tenv t v Hsat Hv Hproved.
  pose proof (prove_is_int_sound _ _ _ _ _ _ Hsat I Hv Hproved)
    as Himmediate.
  simpl in Himmediate. rewrite eval_prim_unary. simpl.
  rewrite (immediate_not_block statics v Himmediate). reflexivity.
Qed.

(* ======================================================================== *)
(* Aliases, boxed numbers and strings                                       *)
(* ======================================================================== *)

Theorem prove_equals_to_simple_of_kind_sound
  : prove_equals_to_simple_of_kind_sound_statement.
Proof. exact prove_equals_to_simple_of_kind_sound_proof. Qed.

Theorem prove_tagging_of_simple_sound
  : prove_tagging_of_simple_sound_statement.
Proof. exact prove_tagging_of_simple_sound_proof. Qed.

Theorem meet_tagging_of_simple_sound : meet_tagging_of_simple_sound_statement.
Proof. exact meet_tagging_of_simple_sound_proof. Qed.

Theorem prove_is_a_boxed_or_tagged_number_sound
  : prove_is_a_boxed_or_tagged_number_sound_statement.
Proof. exact prove_is_a_boxed_or_tagged_number_sound_proof. Qed.

Theorem prove_is_a_tagged_immediate_sound
  : prove_is_a_tagged_immediate_sound_statement.
Proof. exact prove_is_a_tagged_immediate_sound_proof. Qed.

Theorem prove_is_a_boxed_number_sound
  : prove_is_a_boxed_number_sound_statement.
Proof. exact prove_is_a_boxed_number_sound_proof. Qed.

Theorem prove_is_or_is_not_a_boxed_float_sound
  : prove_is_or_is_not_a_boxed_float_sound_statement.
Proof. exact prove_is_or_is_not_a_boxed_float_sound_proof. Qed.

Theorem meet_boxed_number_containing_simple_sound
  : meet_boxed_number_containing_simple_sound_statement.
Proof. exact meet_boxed_number_containing_simple_sound_proof. Qed.

Theorem unbox_number_defined : unbox_number_defined_statement.
Proof.
  intros statics kind v r. rewrite eval_prim_unary.
  change (eval_unary_primitive statics (Unbox_number kind) v)
    with
      (let* c := boxed_number_of statics v in
       if is_number_of_kind kind c then Some (Value_const c) else None).
  destruct (boxed_number_of statics v) as [c | ] eqn:Hboxed;
    [ | discriminate ].
  simpl. destruct (is_number_of_kind kind c) eqn:Hnumber; [ | discriminate ].
  intros [= <-].
  split; [ | exists c; split; first [ exact Hboxed | reflexivity ] ].
  exists c. split; [ exact Hboxed | ].
  destruct kind, c; try discriminate Hnumber; reflexivity.
Qed.

Theorem prove_strings_sound : prove_strings_sound_statement.
Proof. exact prove_strings_sound_proof. Qed.

Theorem meet_strings_sound : meet_strings_sound_statement.
Proof. exact meet_strings_sound_proof. Qed.

Lemma prover_silent_of_made {A B : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (prover' : Typing_env.t -> flambda_type -> option (Proof_of_property.t B)) :
  (∀ (tenv : Typing_env.t) (t : flambda_type) (y : B),
    prover' tenv t = Some (Proof_of_property.Proved y) ->
    ∃ x : A, prover tenv t = Some (Proof_of_property.Proved x)) ->
  prover_silent_on_poison prover ->
  prover_silent_on_poison prover'.
Proof.
  intros Hmade Hsilent statics valuation tenv t y v Hsat Hv Hpoison Hproved.
  destruct (Hmade _ _ _ Hproved) as [x Hx].
  exact (Hsilent _ _ _ _ _ _ Hsat Hv Hpoison Hx).
Qed.

Theorem more_provers_silent_on_poison
  : more_provers_silent_on_poison_statement.
Proof.
  assert (Hnumber :
    prover_silent_on_poison Provers.prove_is_a_boxed_or_tagged_number).
  { exact (gen_value_to_proof_silent _). }
  repeat split.
  - intros min_name_mode. exact (gen_value_to_proof_silent _).
  - intros min_name_mode. exact (gen_value_to_meet_silent _).
  - exact Hnumber.
  - apply (prover_silent_of_made Provers.prove_is_a_boxed_or_tagged_number);
      [ | exact Hnumber ].
    intros tenv t y. unfold Provers.prove_is_a_tagged_immediate.
    destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
      as [[x | ] | ];
      [ intros _; exists x; reflexivity | discriminate | discriminate ].
  - intros kind.
    apply (prover_silent_of_made Provers.prove_is_a_boxed_or_tagged_number);
      [ | exact Hnumber ].
    intros tenv t y. unfold Provers.prove_is_a_boxed_number.
    destruct (Provers.prove_is_a_boxed_or_tagged_number tenv t)
      as [[x | ] | ];
      [ intros _; exists x; reflexivity | discriminate | discriminate ].
  - exact (gen_value_to_proof_silent _).
  - intros kind min_name_mode.
    exact (meet_boxed_number_containing_simple_silent _ _).
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
Qed.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Theorem prove_variant_like_sound : prove_variant_like_sound_statement.
Proof. exact prove_variant_like_sound_proof. Qed.

Theorem meet_variant_like_sound : meet_variant_like_sound_statement.
Proof. exact meet_variant_like_sound_proof. Qed.

Theorem prove_unique_tag_and_size_sound
  : prove_unique_tag_and_size_sound_statement.
Proof. exact prove_unique_tag_and_size_sound_proof. Qed.

Theorem prove_unique_immutable_heap_block_sound
  : prove_unique_immutable_heap_block_sound_statement.
Proof. exact prove_unique_immutable_heap_block_sound_proof. Qed.

Theorem prove_unique_immutable_heap_block_fields
  : prove_unique_immutable_heap_block_fields_statement.
Proof. exact prove_unique_immutable_heap_block_fields_proof. Qed.

Theorem meet_block_field_simple_sound
  : meet_block_field_simple_sound_statement.
Proof. exact meet_block_field_simple_sound_proof. Qed.

Theorem block_load_defined : block_load_defined_statement.
Proof.
  intros statics access_kind mut field v r. rewrite eval_prim_unary. simpl.
  unfold eval_block_load.
  destruct access_kind as [tag size field_kind].
  destruct (block_of statics v) as [[block_tag fields] | ] eqn:Hblock;
    [ | discriminate ].
  destruct
    (known_is tag block_tag && known_is size (Z.of_nat (length fields))
     && (0 <=? field)%Z)
    eqn:Hcondition;
    [ | discriminate ].
  apply andb_prop in Hcondition. destruct Hcondition as [_ Hfield].
  destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
    [ | discriminate ].
  assert (Hloaded :
    kind_of_value loaded = Value ->
    (0 <= field)%Z
    ∧ block_field_loads Value (Z.to_nat field) statics v
    ∧ ∃ (tag' : Z) (fields' : list value),
        block_of statics v = Some (tag', fields')
        ∧ nth_error fields' (Z.to_nat field) = Some loaded).
  { intros Hkind. split; [ exact (proj1 (Z.leb_le _ _) Hfield) | ].
    split.
    - exists block_tag, fields, loaded. auto.
    - exists block_tag, fields. auto. }
  destruct field_kind.
  - destruct (kind_eqb (kind_of_value loaded) Value) eqn:Hkind;
      [ | discriminate ].
    intros [= <-]. rewrite <- Hblock. apply Hloaded.
    exact (proj1 (kind_eqb_eq _ _) Hkind).
  - destruct (is_tagged_immediate loaded) eqn:Htagged; [ | discriminate ].
    intros [= <-]. rewrite <- Hblock. apply Hloaded.
    exact (is_tagged_immediate_kind _ Htagged).
Qed.

Theorem block_provers_silent_on_poison
  : block_provers_silent_on_poison_statement.
Proof.
  assert (Hunique :
    prover_silent_on_poison
      (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value)).
  { exact (gen_value_to_proof_silent _). }
  repeat split.
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
  - apply
      (prover_silent_of_made
         (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value));
      [ | exact Hunique ].
    intros tenv t y. unfold Provers.prove_unique_tag_and_size.
    destruct
      (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value
         tenv t)
      as [[x | ] | ];
      [ intros _; exists x; reflexivity | discriminate | discriminate ].
  - apply
      (prover_silent_of_made
         (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value));
      [ | exact Hunique ].
    intros tenv t y.
    unfold Provers.prove_unique_fully_constructed_immutable_heap_block.
    destruct
      (Provers.gen_value_to_proof Provers.prove_unique_tag_and_size_value
         tenv t)
      as [[x | ] | ];
      [ intros _; exists x; reflexivity | discriminate | discriminate ].
  - intros min_name_mode field_kind field_index.
    exact (gen_value_to_meet_silent _).
Qed.

Theorem prove_single_closures_entry_sound
  : prove_single_closures_entry_sound_statement.
Proof. exact prove_single_closures_entry_sound_proof. Qed.

Theorem meet_single_closures_entry_sound
  : meet_single_closures_entry_sound_statement.
Proof. exact meet_single_closures_entry_sound_proof. Qed.

Theorem single_closures_entry_code : single_closures_entry_code_statement.
Proof.
  intros statics valuation slot alloc_mode entry function_type v
    [functions [value_slots [Hclosure [_ Hcode]]]].
  exists value_slots. apply closure_in_set_of. exists slot, functions.
  split; [ exact Hclosure | exact Hcode ].
Qed.

Theorem prove_code_ids_sound : prove_code_ids_sound_statement.
Proof. exact prove_code_ids_sound_proof. Qed.

Theorem meet_code_ids_sound : meet_code_ids_sound_statement.
Proof. exact meet_code_ids_sound_proof. Qed.

Theorem meet_project_function_slot_simple_sound
  : meet_project_function_slot_simple_sound_statement.
Proof. exact meet_project_function_slot_simple_sound_proof. Qed.

Theorem project_function_slot_in_domain
  : project_function_slot_in_domain_statement.
Proof.
  intros statics move_from move_to v closure Hproject.
  exists move_from, closure. exact Hproject.
Qed.

Theorem project_value_slot_in_domain
  : project_value_slot_in_domain_statement.
Proof.
  intros statics project_from slot value_slot_kind v contents Hproject Hkind.
  exists project_from, contents. split; [ exact Hproject | exact Hkind ].
Qed.

Theorem meet_project_value_slot_simple_sound
  : meet_project_value_slot_simple_sound_statement.
Proof. exact meet_project_value_slot_simple_sound_proof. Qed.

Theorem meet_project_value_slot_simple_known
  : meet_project_value_slot_simple_known_statement.
Proof. exact meet_project_value_slot_simple_known_proof. Qed.

Theorem closure_provers_silent_on_poison
  : closure_provers_silent_on_poison_statement.
Proof.
  repeat split.
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
  - exact (gen_value_to_proof_silent _).
  - exact (gen_value_to_meet_silent _).
  - intros min_name_mode move_to.
    exact (gen_value_to_meet_silent _).
  - intros min_name_mode slot value_slot_kind.
    exact (gen_value_to_meet_silent _).
Qed.

Theorem prove_physical_equality_sound
  : prove_physical_equality_sound_statement.
Proof. exact prove_physical_equality_sound_proof. Qed.

(* A value that [Phys_equal] reads as a tagged immediate is that
   constant. *)
Lemma tagged_immediate_of_int (v : value) (a : Z) :
  int_of_value Standard_tagged_immediate v = Some a ->
  v = Value_const (Tagged_immediate a).
Proof.
  destruct v; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  simpl. destruct (in_range _ _); [ | discriminate ].
  intros H. inversion H. reflexivity.
Qed.

(* Where [Phys_equal] has a value, it is whether the two are the same
   value. *)
Lemma eval_phys_equal_same (statics : static_env) (v1 v2 : value)
    (same : bool) :
  eval_phys_equal statics v1 v2 = Some same -> (same = true <-> v1 = v2).
Proof.
  unfold eval_phys_equal.
  destruct (int_of_value Standard_tagged_immediate v1) as [a | ] eqn:H1;
    destruct (int_of_value Standard_tagged_immediate v2) as [b | ] eqn:H2.
  - intros H. inversion H. apply tagged_immediate_of_int in H1, H2.
    subst v1 v2. rewrite Z.eqb_eq.
    split; [ intros ->; reflexivity | intros Heq; inversion Heq; reflexivity ].
  - destruct (is_allocated statics v2 || is_imported_pointer statics v2);
      [ | discriminate ].
    intros H. inversion H. split; [ discriminate | ].
    intros Heq. subst v2. rewrite H1 in H2. discriminate.
  - destruct (is_allocated statics v1 || is_imported_pointer statics v1);
      [ | discriminate ].
    intros H. inversion H. split; [ discriminate | ].
    intros Heq. subst v2. rewrite H1 in H2. discriminate.
  - destruct (identity_of statics v1) as [s1 | ] eqn:Hi1,
      (identity_of statics v2) as [s2 | ] eqn:Hi2.
    + apply (proj1 (identity_of_imported _ _ _)) in Hi1.
      apply (proj1 (identity_of_imported _ _ _)) in Hi2.
      apply imported_of_symbol in Hi1 as [-> _].
      apply imported_of_symbol in Hi2 as [-> _].
      intros H. inversion H. rewrite Pos.eqb_eq.
      split; [ intros ->; reflexivity | intros Heq; inversion Heq ].
      reflexivity.
    + destruct (is_allocated statics v2 || is_imported_pointer statics v2);
        [ | discriminate ].
      intros H. inversion H. split; [ discriminate | ].
      intros Heq. subst v2. rewrite Hi1 in Hi2. discriminate.
    + destruct (is_allocated statics v1 || is_imported_pointer statics v1);
        [ | discriminate ].
      intros H. inversion H. split; [ discriminate | ].
      intros Heq. subst v2. rewrite Hi1 in Hi2. discriminate.
    + discriminate.
Qed.

Theorem prove_physical_equality_phys_equal :
  prove_physical_equality_phys_equal_statement.
Proof.
  intros statics valuation tenv t1 t2 b op v1 v2 r Hsat Hwf1 Hwf2 Hv1 Hv2
    Hprove Heval.
  pose proof
    (prove_physical_equality_sound statics valuation tenv t1 t2 b v1 v2 Hsat
       Hwf1 Hwf2 Hv1 Hv2 Hprove)
    as Hsound.
  rewrite eval_prim_binary in Heval. simpl in Heval.
  destruct (eval_phys_equal statics v1 v2) as [same | ] eqn:Hsame;
    [ | discriminate ].
  inversion Heval. apply eval_phys_equal_same in Hsame.
  assert (Hb : same = b).
  { destruct b, same; try reflexivity.
    - apply (proj2 Hsame), Hsound.
    - exfalso. apply Hsound, (proj1 Hsame). reflexivity. }
  rewrite Hb. reflexivity.
Qed.

Theorem sound_where_no_poison : sound_where_no_poison_statement.
Proof.
  split.
  - intros A prover input holds Hsound statics valuation tenv t x v Hsat Hno.
    exact
      (Hsound statics valuation tenv t x v
         (satisfied_no_poison_strictly _ _ _ Hno Hsat)).
  - intros A prover input domain holds Hsound statics valuation tenv t v Hsat
      Hno.
    exact
      (Hsound statics valuation tenv t v
         (satisfied_no_poison_strictly _ _ _ Hno Hsat)).
Qed.
